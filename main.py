
import json, io, re, requests
import pandas as pd
import streamlit as st

SHOW_DEBUG = False


st.set_page_config(
    page_title="Quote Process Tracker",
    layout="wide"
)
st.title("Quote Process Tracker")

MONDAY_API_URL = "https://api.monday.com/v2"
# ====== CONFIG ======
api_token = st.text_input("Please insert your Monday API token", type="password", help="Profile → Developer → API token")

# Parent “customers/items” board to browse
BOARD_URL_HARDCODED = "https://coretigo.monday.com/boards/493711664"

# The **subitems board** relation column title that links to products
RELATION_COL_TITLE = "Product Name"


# ====== GraphQL ======
ME_Q = "query { me { id name email } }"

BOARD_GROUPS_Q = """
query BoardGroups($board_id: [ID!]) {
  boards(ids: $board_id) {
    id
    name
    groups { id title }
  }
}
"""

BOARD_SKELETON_Q = """
query BoardSkeleton($board_id: [ID!], $limit: Int!, $cursor: String) {
  boards(ids: $board_id) {
    items_page(limit: $limit, cursor: $cursor) {
      cursor
      items { id name group { title } }
    }
  }
}
"""

SUBITEMS_FOR_ITEM_Q = """
query SubitemsForItem($item_id: [ID!]) {
  items(ids: $item_id) {
    id
    name
    subitems {
      id
      name
      board {
        id
        name
        columns { id title type settings_str }
      }
      column_values { id text value }
    }
  }
}
"""

# Step 1: only get product ids via relation (no product columns here)
SUBITEMS_PRODUCTS_Q = """
query SubitemsProducts($sub_ids:[ID!], $relation_col_id: [String!]) {
  items(ids: $sub_ids) {
    id
    column_values(ids: $relation_col_id) {
      id
      ... on BoardRelationValue {
        linked_item_ids
      }
    }
  }
}
"""

# Get the price-list item ids linked to products (via the specific relation column on the Product board)
PRODUCT_TO_PRICE_IDS_Q = """
query ProductToPrice($prod_ids:[ID!], $price_rel_col_id:[String!]) {
  items(ids:$prod_ids) {
    id
    name
    column_values(ids: $price_rel_col_id) {
      id
      ... on BoardRelationValue {
        linked_item_ids
      }
    }
  }
}
"""

# Fetch price values directly from the Price List board items
PRICE_ITEMS_FETCH_Q = """
query PriceItems($price_ids:[ID!], $price_col_ids:[String!]) {
  items(ids: $price_ids) {
    id
    name
    column_values(ids: $price_col_ids) {
      id
      text
      value
      type
    }
  }
}
"""

# Step 2: fetch requested product columns by querying product items directly
PRODUCTS_BY_IDS_Q = """
query ProductsByIds($ids:[ID!], $prod_col_ids:[String!]) {
  items(ids: $ids) {
    id
    name
    column_values(ids: $prod_col_ids) {
      id
      text
      value
      type
    }
  }
}
"""

PRODUCT_BOARD_META_Q = """
query ProductBoardMeta($board_id: [ID!]) {
  boards(ids: $board_id) {
    id
    name
    columns { id title type }
  }
}
"""

PRODUCTS_BY_IDS_ALL_Q = """
query ProductsByIdsAll($ids:[ID!]) {
  items(ids: $ids) {
    id
    name
    column_values { id text value type }
  }
}
"""

# ====== Helpers ======
import re

# ==== SIMPLE NAME MATCHING (exactly as requested) ====
# normalize: lowercase + spaces -> underscores; keep symbols like $ / €
import re

def _norm_simple(s) -> str:
    s = "" if s is None else str(s)
    return re.sub(r"\s+", "_", s.strip().lower())

def _swap_last_token(normed: str) -> str:
    parts = normed.split("_")
    return normed if len(parts) <= 1 else "_".join([parts[-1]] + parts[:-1])

def _variants_simple(title: str) -> tuple[str, str]:
    """
    Two forms only:
      1) normalized
      2) last-token-first permutation (e.g., list_price_$  <->  $_list_price)
    """
    n = _norm_simple(title)
    return n, _swap_last_token(n)

def best_name_match_simple(target_title: str, candidate_titles: list[str]) -> str:
    """
    Match if any of these are equal:
      - norm(target) == norm(candidate)
      - swap(target) == norm(candidate)
      - norm(target) == swap(candidate)
    """
    t_norm, t_swap = _variants_simple(target_title)
    for c in candidate_titles:
        c_norm, c_swap = _variants_simple(c)
        if t_norm == c_norm or t_swap == c_norm or t_norm == c_swap:
            return c
    return None
# ==== END SIMPLE NAME MATCHING ====

def price_backed_product_titles(token: str, product_board_id: str, price_rel_col_id: str) -> set[str]:
    """
    Return the set of Product-board column *titles* that are mirror/lookup columns
    fed by the given Product→Price relation column.
    """
    schema = gql(token, """
      query($bid:[ID!]) {
        boards(ids:$bid){
          id
          columns { id title type settings_str }
        }
      }
    """, {"bid": [product_board_id]})

    cols = ((schema.get("boards") or [{}])[0].get("columns")) or []
    backed = set()
    for c in cols:
        if (c.get("type") or "").lower() not in ("mirror", "lookup"):
            continue
        try:
            s = json.loads(c.get("settings_str") or "{}")
        except Exception:
            s = {}

        rel_ok = False
        # common encodings across tenants
        if s.get("relationColumnId") == price_rel_col_id:
            rel_ok = True
        if price_rel_col_id in (s.get("relationColumns") or []):
            rel_ok = True
        for lc in (s.get("linkedColumns") or []):
            if isinstance(lc, dict) and lc.get("columnId") == price_rel_col_id:
                rel_ok = True
                break

        if rel_ok:
            t = (c.get("title") or "").strip()
            if t:
                backed.add(t)
    return backed

import re

def _title_norm(t: str):
    """Return (normalized_base, currency) for a column title."""
    if not t:
        return "", None
    s = t.casefold().strip()

    # detect currency
    cur = None
    if "$" in s or " usd" in s or "($" in s:
        cur = "$"
    elif "€" in s or " eur" in s or "(€" in s or "euro" in s:
        cur = "€"

    # strip currency chars & parenthetical suffixes for base comparison
    s = s.replace("$", "").replace("€", "")
    s = re.sub(r"\(.*?\)", "", s)          # remove text in parentheses
    s = re.sub(r"[^a-z0-9]+", "", s)       # keep alnum, drop spaces/punct
    return s, cur

def build_desired_to_actual_map(actual_titles: list[str], desired_titles: list[str]) -> dict[str, str]:
    """
    Map each desired title to the best matching actual title by normalized base,
    preferring the same currency when present.
    """
    # index actual titles
    actual_norm = []
    for a in actual_titles:
        nb, cur = _title_norm(a)
        actual_norm.append((a, nb, cur))

    out = {}
    for d in desired_titles:
        if d == "Product Name":
            out[d] = "Product Name"  # handled specially anyway
            continue

        dnb, dcur = _title_norm(d)

        # 1) exact normalized base + same currency
        candidates = [a for (a, nb, cur) in actual_norm if nb == dnb and ((dcur is None) or (cur == dcur))]
        if not candidates:
            # 2) exact normalized base (currency ignored)
            candidates = [a for (a, nb, _cur) in actual_norm if nb == dnb]
        if not candidates:
            # 3) loose contains match on raw strings (last resort)
            candidates = [a for a in actual_titles if d.casefold().replace(" ", "") in a.casefold().replace(" ", "")]

        if candidates:
            out[d] = candidates[0]  # pick first best
        # else: no mapping; leave unmapped
    return out


def find_price_list_relation_on_product_board(token: str, product_board_id: str) -> dict:
    """
    Return the *first* board_relation column on the Product board that points to another board
    (your Price List). If you have multiple, you can refine by title contains 'Price'.
    """
    data = gql(token, PRODUCT_BOARD_META_Q, {"board_id": [product_board_id]})
    boards = data.get("boards") or []
    if not boards:
        return None
    cols = boards[0].get("columns") or []

    # Prefer a relation that looks like price mapping
    for c in cols:
        if c.get("type") == "board_relation":
            try:
                s = json.loads(c.get("settings_str") or "{}")
            except Exception:
                s = {}
            title = (c.get("title") or "").lower()
            if "price" in title or "list" in title or "cost" in title:
                return {"id": c["id"], "title": c.get("title"), "settings": s}

    # fallback: first relation
    for c in cols:
        if c.get("type") == "board_relation":
            try:
                s = json.loads(c.get("settings_str") or "{}")
            except Exception:
                s = {}
            return {"id": c["id"], "title": c.get("title"), "settings": s}

    return None


def product_board_id_from_subitems(subitems: list) -> str :
    """You already have this implicitly; here's a quick extractor."""
    if not subitems:
        return None
    b = (subitems[0].get("board") or {})
    return str(b.get("id")) if b.get("id") else None


def map_products_to_price_items(token: str, product_ids: list[str], price_rel_col_id: str) -> dict[str, str]:
    """
    Return {product_id -> price_item_id} (first linked price item).
    """
    data = gql(token, PRODUCT_TO_PRICE_IDS_Q, {
        "prod_ids": product_ids,
        "price_rel_col_id": [price_rel_col_id]
    })
    mapping = {}
    for it in (data.get("items") or []):
        pid = str(it["id"])
        cv = (it.get("column_values") or [{}])[0]
        linked = (cv or {}).get("linked_item_ids") or []
        if linked:
            mapping[pid] = str(linked[0])
    return mapping


def build_price_col_id_map(token: str, price_board_id: str) -> dict[str, str]:
    data = gql(token, PRODUCT_BOARD_META_Q, {"board_id": [price_board_id]})
    boards = data.get("boards") or []
    cols = boards[0].get("columns") or [] if boards else []
    return {(c.get("title") or "").strip(): c["id"] for c in cols if (c.get("title") or "").strip()}





def fetch_price_values(token: str, price_item_ids: list[str], price_cols_map: dict[str,str]) -> dict[str, dict]:
    """
    Return {price_item_id: {Title->value}}
    """
    if not price_item_ids:
        return {}
    data = gql(token, PRICE_ITEMS_FETCH_Q, {
        "price_ids": price_item_ids,
        "price_col_ids": list(price_cols_map.values())
    })
    out = {}
    for it in (data.get("items") or []):
        rid = str(it["id"])
        row = {}
        cv_by_id = {cv["id"]: cv for cv in (it.get("column_values") or [])}
        for title, cid in price_cols_map.items():
            row[title] = _cell_text(cv_by_id.get(cid))
        out[rid] = row
    return out


def gql(token: str, query: str, variables: dict = None) -> dict:
    headers = {"Authorization": token or "", "Content-Type": "application/json", "Accept": "application/json"}
    resp = requests.post(MONDAY_API_URL, headers=headers, json={"query": query, "variables": variables or {}}, timeout=60)
    payload = None
    try:
        payload = resp.json()
    except Exception:
        pass

    if not resp.ok or not payload:
        st.error(f"HTTP {resp.status_code} from Monday API")
        st.code((resp.text or "")[:1800], language="json")
        raise RuntimeError(f"HTTP error {resp.status_code}")

    if "errors" in payload:
        st.error("GraphQL errors from Monday API")
        st.code(json.dumps(payload["errors"], indent=2)[:2000], language="json")
        raise RuntimeError("GraphQL returned errors")

    data = payload.get("data") or {}
    return data

def extract_board_id(raw: str) -> str:
    if not raw: return None
    m = re.search(r"/boards/(\d+)", raw.strip())
    if m: return m.group(1)
    if raw.strip().isdigit(): return raw.strip()
    m = re.search(r"\d{6,}", raw.strip())
    return m.group(0) if m else None

def _dedupe_titles(cols):
    seen, out = {}, []
    for c in cols or []:
        t = (c.get("title") or c["id"]).strip() or c["id"]
        if t in seen:
            seen[t] += 1
            t = f"{t} ({seen[t]})"
        else:
            seen[t] = 1
        out.append({"id": c["id"], "title": t, "type": c.get("type"), "settings_str": c.get("settings_str")})
    return out

def _cell_text(cell: dict = None) -> str:
    """Robust human-facing text from Monday column value payload."""
    if not cell:
        return None
    t = cell.get("text")
    if t not in (None, ""):
        return t
    val = cell.get("value")
    if val in (None, ""):
        return None
    try:
        j = json.loads(val)
    except json.JSONDecodeError:
        return str(val)

    if isinstance(j, (int, float)):
        return str(j)
    if isinstance(j, str):
        return j
    if isinstance(j, dict):
        for k in ("display_value", "text", "plain_text", "value", "content", "number", "amount"):
            v = j.get(k)
            if isinstance(v, (str, int, float)):
                return str(v)
    return None

def _empty_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([True])
    m = s.isna()
    if s.dtype == object:
        m = m | s.astype(str).str.strip().isin(["", "None", "nan", "NaN"])
    return m

@st.cache_data(show_spinner=False)
def load_board_groups(token: str, board_id: str):
    data = gql(token, BOARD_GROUPS_Q, {"board_id": [board_id]})
    b = (data.get("boards") or [None])[0]
    if not b:
        return {"id": board_id, "name": ""}, []
    return {"id": b["id"], "name": b["name"]}, b["groups"]

@st.cache_data(show_spinner=False)
def collect_parent_items_in_group(token: str, board_id: str, target_group: str, page_size: int = 500):
    cursor, items = None, []
    tgt = (target_group or "").strip().casefold()
    while True:
        data = gql(token, BOARD_SKELETON_Q, {"board_id": [board_id], "limit": page_size, "cursor": cursor})
        b = (data.get("boards") or [None])[0]
        if not b: break
        page = b["items_page"]
        for it in (page.get("items") or []):
            grp = (it.get("group") or {}).get("title") or ""
            if grp.strip().casefold() == tgt:
                items.append({"id": str(it["id"]), "name": it["name"]})
        cursor = page.get("cursor")
        if not cursor: break
    items.sort(key=lambda r: (r["name"] or "").casefold())
    return items

def fetch_subitems_and_columns(token: str, parent_item_id: str):
    data = gql(token, SUBITEMS_FOR_ITEM_Q, {"item_id": [str(parent_item_id)]})
    items = data.get("items") or []
    if not items:
        return [], [], None, ""
    subitems = items[0].get("subitems") or []
    if not subitems:
        return [], [], None, ""
    cols = _dedupe_titles((subitems[0].get("board") or {}).get("columns") or [])
    sub_board = subitems[0].get("board") or {}
    return subitems, cols, str(sub_board.get("id")), sub_board.get("name") or ""

def flatten_subitems(subitems, columns_meta):
    rows = []
    for si in subitems:
        row = {"subitem_id": str(si["id"]), "Subitem (name)": si["name"]}
        cv_by_id = {cv["id"]: cv for cv in (si.get("column_values") or [])}
        for col in columns_meta:
            row[col["title"]] = _cell_text(cv_by_id.get(col["id"]))
        rows.append(row)
    rows.sort(key=lambda r: (r.get("Subitem (name)") or "").casefold())
    return pd.DataFrame(rows)

def find_relation_col(columns_meta, title_guess: str) -> dict:
    tnorm = (title_guess or "").strip().casefold()
    for c in columns_meta or []:
        if (c.get("type") == "board_relation") and ((c.get("title") or "").strip().casefold() == tnorm):
            return c
    for c in columns_meta or []:
        if c.get("type") == "board_relation":
            return c
    return None

def product_board_id_from_relation(col: dict) -> str:
    try:
        s = json.loads(col.get("settings_str") or "{}")
        ids = s.get("boardIds") or []
        if ids:
            return str(ids[0])
    except Exception:
        pass
    return None

def build_product_col_id_map(token: str, product_board_id: str) -> dict[str, str]:
    """Return {title -> column_id} for the product board."""
    data = gql(token, PRODUCT_BOARD_META_Q, {"board_id": [product_board_id]})
    boards = data.get("boards") or []
    if not boards: return {}
    cols = boards[0].get("columns") or []
    out = {}
    for c in cols:
        title = (c.get("title") or "").strip()
        if title:
            out[title] = c["id"]
    return out

def fetch_linked_product_values_two_step(
    token: str,
    subitems: list,
    relation_col_id: str,
    product_col_id_map: dict[str, str]
) -> dict[str, dict]:
    """
    Return {subitem_id: {Desired Title -> value, "product_id" -> id, "Product Name" -> name}}.

    Step A: subitem -> product ids (BoardRelationValue)
    Step B: product ids -> fetch product columns directly (works for mirrors/prices)
    """
    sub_ids = [str(si["id"]) for si in subitems]

    # ---- Step A: subitem -> product ids
    data_rel = gql(
        token,
        SUBITEMS_PRODUCTS_Q,
        {"sub_ids": sub_ids, "relation_col_id": [relation_col_id]},
    )

    sub_to_prod = {}
    unique_prod_ids = set()
    for it in (data_rel.get("items") or []):
        sid = str(it["id"])
        cv = next((cv for cv in (it.get("column_values") or []) if cv["id"] == relation_col_id), None)
        pids = (cv or {}).get("linked_item_ids") or []
        if pids:
            pid = str(pids[0])           # assume first linked product
            sub_to_prod[sid] = pid
            unique_prod_ids.add(pid)

    # Map the requested product columns → ids (skip Product Name: it is the item name)
    # Fetch ALL product columns (no filter)
    data_prod = gql(token, PRODUCTS_BY_IDS_ALL_Q, {"ids": list(unique_prod_ids)})

    # Build id->title map from the already-loaded product schema
    # (you already have product_col_id_map: {title -> id})
    title_by_id = {cid: title for title, cid in product_col_id_map.items()}

    prod_map = {}
    for pit in (data_prod.get("items") or []):
        pid = str(pit["id"])
        row = {"Product Name": pit.get("name")}
        for cv in (pit.get("column_values") or []):
            title = title_by_id.get(cv["id"]) or cv["id"]  # fall back to id if title unknown
            row[title] = _cell_text(cv)
        prod_map[pid] = row


    # Now prepare subitem mapping using sub_to_prod + prod_map
    out = {}
    for sid in sub_ids:
        pid = sub_to_prod.get(sid)
        row = {"product_id": pid}
        if pid and pid in prod_map:
            row.update(prod_map[pid])
        out[sid] = row

    # Small debug
    if SHOW_DEBUG:
        with st.expander("🔧 Debug: product values fetched (first rows)", expanded=False):
            preview = []
            for sid, row in list(out.items())[:5]:
                r = {"subitem_id": sid, **row}
                preview.append(r)
            st.json(preview)


    return out



# ---------- NEW: Stage 2.5 price enrichment from Price List (if Product board links to it) ----------
def enrich_linked_values_with_price_list(token, product_board_id, linked_values):
    if not linked_values or not product_board_id:
        return linked_values

    price_rel = find_price_list_relation_on_product_board(token, product_board_id)
    if not price_rel:
        return linked_values

    board_ids = (price_rel.get("settings") or {}).get("boardIds") or []
    if not board_ids:
        return linked_values
    price_board_id = str(board_ids[0])

    # Titles of product columns that *should* be sourced from the Price board
    backed_titles = price_backed_product_titles(token, product_board_id, price_rel["id"])

    # Build a map of ALL price-board columns (Title -> id) and fetch rows
    price_cols_map = build_price_col_id_map(token, price_board_id)
    if not price_cols_map:
        return linked_values

    product_ids = sorted({v.get("product_id") for v in linked_values.values() if v.get("product_id")})
    if not product_ids:
        return linked_values

    prod_to_price = map_products_to_price_items(token, product_ids, price_rel["id"])
    if not prod_to_price:
        return linked_values

    price_item_ids = sorted(set(prod_to_price.values()))
    price_rows = fetch_price_values(token, price_item_ids, price_cols_map)

    # Overlay only for product columns known to be backed by the Price board
    for sid, vals in linked_values.items():
        pid = vals.get("product_id")
        price_id = prod_to_price.get(pid) if pid else None
        pv = price_rows.get(price_id) or {}
        for title in backed_titles:
            v_price = pv.get(title)
            if v_price not in (None, "", "None"):
                vals[title] = v_price

    # Optional debug of raw price rows
    if SHOW_DEBUG:
        with st.expander("🔧 Debug: Price List source rows (ALL columns)", expanded=False):
            rows = [{"price_item_id": rid, **cols} for rid, cols in price_rows.items()]
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.write("No price rows fetched.")

    return linked_values




def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    try:
        with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
            for name, d in sheets.items():
                d.to_excel(xw, sheet_name=name[:31], index=False)
    except ModuleNotFoundError:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as xw:
            for name, d in sheets.items():
                d.to_excel(xw, sheet_name=name[:31], index=False)
    bio.seek(0)
    return bio.getvalue()

def as_number(s):
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    txt = str(s)
    txt = re.sub(r"[^\d\.\-\,]", "", txt)
    if txt.count(",") and txt.count("."):
        txt = txt.replace(",", "")
    else:
        txt = txt.replace(",", ".")
    try:
        return float(txt)
    except Exception:
        return None

# ====== STEP 0: Auth + load groups ======
if st.button("Load customers (tables)", type="primary"):
    if not api_token or not api_token.strip():
        st.error("Please paste your Monday API token.")
        st.stop()

    board_id = extract_board_id(BOARD_URL_HARDCODED)
    if not board_id:
        st.error("Hard-coded board URL is invalid.")
        st.stop()

    with st.spinner("Authenticating…"):
        me = gql(api_token, ME_Q)
    st.success(f"Authenticated as {me['me']['name']} ({me['me']['email']})")

    with st.spinner("Loading groups…"):
        meta, groups = load_board_groups(api_token, board_id)

    if not groups:
        st.warning("No groups found on this board.")
        st.stop()

    st.session_state["_board_id"] = board_id
    st.session_state["_board_meta"] = meta
    st.session_state["_group_titles"] = sorted((g["title"] for g in groups), key=str.casefold)

    # clear
    for k in ["_items_for_group", "_picked_group", "_subitems", "_sub_cols", "_sub_df",
              "_relation_col", "_product_board_id", "_prod_col_map", "_linked_values"]:
        st.session_state.pop(k, None)

# ====== STEP 1: choose group, load items ======
if "_group_titles" in st.session_state:
    st.write("")
    picked_group = st.selectbox(
        "Choose a customer:",
        options=st.session_state["_group_titles"],
        index=None,
        placeholder="Select…",
        key="group_pick",
    )

    if st.button("Load items of this customer") and picked_group:
        with st.spinner("Collecting items…"):
            items = collect_parent_items_in_group(api_token, st.session_state["_board_id"], picked_group, page_size=500)
        if not items:
            st.warning("No items found in that customer.")
            st.stop()
        st.session_state["_items_for_group"] = items
        st.session_state["_picked_group"] = picked_group
        # clear downstream
        for k in ["_subitems", "_sub_cols", "_sub_df", "_relation_col", "_product_board_id", "_prod_col_map", "_linked_values"]:
            st.session_state.pop(k, None)

# ====== STEP 2: choose item, fetch subitems & relation/product meta ======
if "_items_for_group" in st.session_state:
    st.write("")
    item_label = st.selectbox(
        f"Choose an item under “{st.session_state.get('_picked_group','')}”:",
        options=[f"{it['name']}  (#{it['id']})" for it in st.session_state["_items_for_group"]],
        index=None,
        placeholder="Select…",
        key="item_pick",
    )
    parent_item_id = None
    if item_label:
        parent_item_id = item_label.rsplit("(#", 1)[-1].rstrip(")")

    st.write("")
    st.write("")
    if st.button("Open subitems") and parent_item_id:
        with st.spinner("Fetching subitems…"):
            subitems, sub_cols, sub_board_id, _ = fetch_subitems_and_columns(api_token, parent_item_id)
        if not subitems:
            st.warning("No subitems found for that item.")
            st.stop()

        st.session_state["_subitems"] = subitems
        st.session_state["_sub_cols"] = sub_cols
        st.session_state["_sub_df"] = flatten_subitems(subitems, sub_cols)

        # relation column on subitems board
        rel_col = find_relation_col(sub_cols, RELATION_COL_TITLE)
        if not rel_col:
            st.error(f"Could not find a board relation column (looked for “{RELATION_COL_TITLE}”).")
            st.stop()
        st.session_state["_relation_col"] = rel_col

        # which product board does that relation point to?
        pbid = product_board_id_from_relation(rel_col)
        if not pbid:
            st.error("Could not deduce the product board id from relation settings.")
            st.stop()
        st.session_state["_product_board_id"] = pbid

        with st.spinner("Loading product board metadata…"):
            col_map = build_product_col_id_map(api_token, pbid)
        st.session_state["_prod_col_map"] = col_map

        # === DEBUG: Product board schema + one product item values + relation targets ===
        if SHOW_DEBUG:
            with st.expander("🔎 DEBUG – Product board columns & values (please expand)", expanded=True):
                token = api_token
                prod_board_id = st.session_state["_product_board_id"]
                prod_col_map = st.session_state["_prod_col_map"]  # {title -> id}

                # 1) Product board columns (schema)
                st.subheader("Product board columns (schema)")
                # Re-fetch with types+settings so we can inspect relations/mirrors
                schema = gql(token, """
                query($bid:[ID!]) {
                  boards(ids:$bid){
                    id name
                    columns { id title type settings_str }
                  }
                }
                """, {"bid": [prod_board_id]})
                pcols = (schema.get("boards") or [{}])[0].get("columns") or []
                cols_df = pd.DataFrame([{
                    "id": c["id"],
                    "title": c.get("title"),
                    "type": c.get("type"),
                    "settings_str": c.get("settings_str")
                } for c in pcols]).sort_values("title", key=lambda s: s.str.lower())
                st.dataframe(cols_df[["id", "title", "type"]], use_container_width=True)

                # Helper: find first linked Product ID from current subitems
                subitems = st.session_state.get("_subitems") or []

        with st.spinner("Resolving linked product values…"):
            linked = fetch_linked_product_values_two_step(
                api_token,
                subitems=subitems,
                relation_col_id=rel_col["id"],
                product_col_id_map=col_map
            )

        # ---------- NEW: Price enrichment (runs in same block to avoid KeyError) ----------
        with st.spinner("Looking for Price List relations and enriching prices…"):
            linked = enrich_linked_values_with_price_list(api_token, product_board_id=pbid, linked_values=linked)
        # ---------- END NEW ----------

        st.session_state["_linked_values"] = linked
        # Clear any prior saved/edited data so the new item will re-init from fresh final_df
        st.session_state.pop("master_df", None)
        st.session_state.pop("_edit_buffer", None)

# ==== FINAL (no refetch, no merge): fill _sub_df from _linked_values by normalized names ====

import re

def _is_blank(v):
    return v is None or str(v).strip() in ("", "None", "nan", "NaN")

def _slug(s: str) -> str:
    s = "" if s is None else str(s)      # <— add this cast
    s = s.strip().lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9 $€£₪]+", " ", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s


def _variants(title: str) -> set[str]:
    """normalized name set: base, base_$, $_base"""
    s = _slug(title or "")
    cur = None
    if "$" in s or "usd" in s: cur = "$"
    elif "€" in s or "eur" in s or "euro" in s: cur = "€"
    elif "£" in s or "gbp" in s or "pound" in s: cur = "£"
    elif "₪" in s or "ils" in s or "nis" in s: cur = "₪"
    base = s.replace("$","").replace("€","").replace("£","").replace("₪","").strip("_")
    out = {base}
    if cur:
        out.add(f"{base}_{cur}")
        out.add(f"{cur}_{base}")
    return out

def _build_matchers(sub_cols: list[str], prod_keys: list[str]):
    """return dicts for fast lookups"""
    sub_norm = {}
    for c in sub_cols:
        for v in _variants(c):
            sub_norm.setdefault(v, set()).add(c)
    prod_norm = {}
    for k in prod_keys:
        for v in _variants(k):
            prod_norm.setdefault(v, set()).add(k)
    return sub_norm, prod_norm

# your explicit pairs (keep these — they win when present)
EXPLICIT = [
    ("$ List Price","List Price $"),
    ("$ List Price","List Price €"),
    ("List Price £","List Price $"),
    ("List Price £","List Price €"),
    ("List Price ₪","List Price $"),
    ("List Price ₪","List Price €"),
    ("€ List Price","List Price $"),
    ("€ List Price","List Price €"),
    ("Cost ($)","CT Cost ($)"),
    ("Cost (€)","CT Cost (€)"),
]

sub_df = st.session_state.get("_sub_df")
linked  = st.session_state.get("_linked_values") or {}

if not (isinstance(sub_df, pd.DataFrame) and not sub_df.empty):
     st.stop()

final_df = sub_df.copy()
# Columns we are allowed to fill into (as plain strings)
sub_col_names = [str(c) for c in final_df.columns if c != "subitem_id"]

#sub_cols = [c for c in final_df.columns if c != "subitem_id"]
for i, row in final_df.iterrows():
    sid = row["subitem_id"]
    pvals = (st.session_state.get("_linked_values") or {}).get(sid, {}) or {}
    if not pvals:
        continue

    # 1) apply any explicit pairs first (if you keep them)
    for prod_title, sub_title in EXPLICIT:
        if prod_title in pvals and sub_title in final_df.columns:
            val = pvals.get(prod_title)
            if val not in (None, "", "None"):
                final_df.at[i, sub_title] = val

    # 2) strict simple matching per your rule
    for pkey, pval in pvals.items():
        if pval in (None, "", "None"):
            continue


        sub_match = best_name_match_simple(str(pkey), sub_col_names)

        if sub_match:
            final_df.at[i, sub_match] = pval

# Precompute normalization maps once (union of all product keys we have)
all_prod_keys = sorted({k for d in linked.values() for k in (d or {}).keys()})
sub_norm, prod_norm = _build_matchers(sub_col_names, all_prod_keys)

applied, missed_prod_keys, missed_sub_cols = [], set(), set()

for i, row in final_df.iterrows():
    sid = row["subitem_id"]
    pvals = linked.get(sid) or {}
    if not pvals:
        continue

    # 1) apply explicit pairs first
    for prod_title, sub_title in EXPLICIT:
        if prod_title in pvals and sub_title in final_df.columns:
            val = pvals.get(prod_title)
            if not _is_blank(val):
                final_df.at[i, sub_title] = val
                applied.append((sid, "explicit", prod_title, "→", sub_title))

    # 2) generic normalized matching: for every product key, find sub columns by variants
    # 2) strict simple matching per your rule (lower+underscores; if no hit, try last-token-first)
    for pkey, pval in pvals.items():
        if pval in (None, "", "None"):
            continue
        # find ONE matching sub column by the simple rule
        sub_match = best_name_match_simple(str(pkey), sub_col_names)
        if sub_match:
            final_df.at[i, sub_match] = pval

# (Optional) quick completeness check
if SHOW_DEBUG:
    with st.expander("🧪 Fill debug", expanded=False):
        st.write("Applied fills:", len(applied))
        if missed_prod_keys:
            st.write("Product keys that never matched any sub column (by normalization):")
            st.write(sorted(missed_prod_keys)[:50])
        # show non-empty counts for a few important columns
        probe = ["Product Name","SKU","Description","List Price $","List Price €","CT Cost ($)","CT Cost (€)","Currency"]
        counts = {c: int(final_df[c].astype(str).str.strip().isin(["","","None","nan","NaN"]).__invert__().sum())
                  for c in probe if c in final_df.columns}
        st.write("Non-empty counts:", counts)

# --- Calculated columns setup -----------------------------------------------
COMPUTED_COLS = [
    "Total List Price ($)",
    "Row Discount Amount ($)",
    "Net Price After Discount ($)",
]

def _col_like(df: pd.DataFrame, title: str) -> str :
    """Best-effort column (exact or your 'simple' matcher)."""
    if title in df.columns:
        return title
    hit = best_name_match_simple(title, list(map(str, df.columns)))
    return hit

def recompute_calculated(df: pd.DataFrame) -> pd.DataFrame:
    """(Re)compute the 3 calculated columns on a copy of df and return it."""
    df = df.copy()

    # resolve source columns by name (tolerant to variants)
    c_qty   = _col_like(df, "Quantity")
    c_list  = _col_like(df, "List Price $")
    c_discP = _col_like(df, "Discount (%)")

    qty   = df[c_qty].map(as_number)   if c_qty   else 0.0
    lpu   = df[c_list].map(as_number)  if c_list  else 0.0
    discP = df[c_discP].map(as_number) if c_discP else 0.0

    total_list = (qty.fillna(0) * lpu.fillna(0))
    row_disc   = total_list * (discP.fillna(0) / 100.0)
    net_after  = total_list - row_disc

    df["Total List Price ($)"]        = total_list
    df["Row Discount Amount ($)"]     = row_disc
    df["Net Price After Discount ($)"] = net_after
    return df
# ---------------------------------------------------------------------------

# ---- initialize the authoritative, editable dataset from the filled table
if "master_df" not in st.session_state:
    st.session_state["master_df"] = recompute_calculated(final_df)


# --- COLUMN PICKER + EXPORT (safe state pattern) ---
def _on_view_changed():
    # force the editor to remount when filters/columns change
    st.session_state.pop("editor_view_df", None)
    st.session_state.pop("_edit_buffer", None)   # ← add this line


with st.container(border=True):
    st.subheader("Columns & Export")

    base_df = st.session_state["master_df"]              # << use the saved table

    # EXCLUDE computed columns from the picker so they can't be dropped
    all_pickable = [c for c in base_df.columns if c not in (["subitem_id"] + COMPUTED_COLS)]

    if "col_pick" not in st.session_state:
        st.session_state["col_pick"] = all_pickable[:]

    chosen = st.multiselect(
        "Columns to show (drag/select to reorder):",
        options=all_pickable,
        key="col_pick",
        help="Unselect columns to drop from the view.",
        on_change=_on_view_changed,  # <— add this
    )

    if st.button("Reset to all columns"):
        st.session_state.pop("col_pick", None)
        st.rerun()

    show_id = st.checkbox("Include subitem_id", value=False)

    # Build the view from the master and ALWAYS append computed columns
    view_cols = (["subitem_id"] if show_id else []) + chosen + COMPUTED_COLS
    # dedupe while preserving order
    seen = set()
    view_cols = [c for c in view_cols if not (c in seen or seen.add(c))]
    view_df = base_df[view_cols].copy()



# --- TABLE (single) : view OR edit the same view_df (live formulas) -----
st.write("")  # small spacing
st.subheader("Subitems table")

# Names of the 3 derived columns (do NOT include in the picker)
DERIVED_COLS = [
    "Total List Price ($)",
    "Row Discount Amount ($)",
    "Net Price After Discount ($)",
]

# Helper: find likely column names in the current view/master
def _need_like(title: str, cols: list[str]) -> str :
    if title in cols:
        return title
    hit = best_name_match_simple(title, cols)
    return hit

# Build the editor dataframe from the authoritative master view (so edits persist per your picker)
edit_base = view_df.copy()

# Compute formulas _before_ rendering so the editor shows fresh values every rerun
all_cols_str = list(map(str, edit_base.columns))
qty_col      = _need_like("Quantity", all_cols_str)
price_col    = _need_like("List Price $", all_cols_str)
disc_pct_col = _need_like("Discount (%)", all_cols_str)

def _num(s):
    return pd.to_numeric(s.apply(as_number), errors="coerce")

qty  = _num(edit_base[qty_col])      if qty_col      in edit_base.columns else pd.Series(0, index=edit_base.index)
lpu  = _num(edit_base[price_col])    if price_col    in edit_base.columns else pd.Series(0, index=edit_base.index)
dpct = _num(edit_base[disc_pct_col]) if disc_pct_col in edit_base.columns else pd.Series(0, index=edit_base.index)

tot_list = (qty * lpu).fillna(0.0)
row_disc = (tot_list * (dpct/100.0)).fillna(0.0)
net_price = (tot_list - row_disc).fillna(0.0)

# Attach/refresh the derived columns (always present, read-only)
edit_base[DERIVED_COLS[0]] = tot_list
edit_base[DERIVED_COLS[1]] = row_disc
edit_base[DERIVED_COLS[2]] = net_price

# Keep subitem_id and derived columns read-only
readonly_cols = [c for c in edit_base.columns if c in DERIVED_COLS or c == "subitem_id"]

edit_mode = st.toggle("✏️ Edit table", value=False, key="edit_toggle", help="Turn on to edit the table below.")

if edit_mode:
    # 1) take the live buffer (or the latest computed view), then compute derived cols BEFORE rendering
    buf = st.session_state.get("_edit_buffer", edit_base).copy()

    cols_str = list(map(str, buf.columns))
    def _need_like(title: str):
        if title in cols_str:
            return title
        return best_name_match_simple(title, cols_str)

    qty_col      = _need_like("Quantity")
    price_col    = _need_like("List Price $")
    disc_pct_col = _need_like("Discount (%)")

    def _num(s: pd.Series):
        return pd.to_numeric(s.apply(as_number), errors="coerce")

    qty  = _num(buf[qty_col])      if qty_col      in buf.columns else pd.Series(0, index=buf.index)
    lpu  = _num(buf[price_col])    if price_col    in buf.columns else pd.Series(0, index=buf.index)
    dpct = _num(buf[disc_pct_col]) if disc_pct_col in buf.columns else pd.Series(0, index=buf.index)

    buf[DERIVED_COLS[0]] = (qty * lpu).fillna(0.0)
    buf[DERIVED_COLS[1]] = (buf[DERIVED_COLS[0]] * (dpct / 100.0)).fillna(0.0)
    buf[DERIVED_COLS[2]] = (buf[DERIVED_COLS[0]] - buf[DERIVED_COLS[1]]).fillna(0.0)

    # 2) render editor; dynamic key so it remounts when visible rows/cols change (fixes filter issue)
    row_ids = tuple(buf["subitem_id"]) if "subitem_id" in buf.columns else tuple(buf.index)
    _editor_fp = (tuple(buf.columns), row_ids)
    editor_key = f"editor_view_df_{hash(_editor_fp)}"

    readonly_cols = [c for c in buf.columns if c in DERIVED_COLS or c == "subitem_id"]

    edited = st.data_editor(
        buf,
        key=editor_key,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        disabled=readonly_cols,
    )

    # 3) keep only editable columns in the live buffer; next rerun will recompute derived columns
    editable_cols = [c for c in edited.columns if c not in (DERIVED_COLS + ["subitem_id"])]
    st.session_state["_edit_buffer"] = edited[editable_cols].copy()

    # Save + Revert
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("💾 Save edits", type="primary", use_container_width=True):
            master = st.session_state["master_df"].set_index("subitem_id") if "subitem_id" in st.session_state["master_df"].columns else st.session_state["master_df"].copy()
            incoming = edited.set_index("subitem_id") if "subitem_id" in edited.columns else edited.copy()

            # Ensure rows match incoming (supports add/remove)
            master = master.reindex(incoming.index)

            updatable = [c for c in editable_cols if c in master.columns]
            master.loc[incoming.index, updatable] = incoming[updatable]

            st.session_state["master_df"] = master.reset_index() if "subitem_id" in st.session_state["master_df"].columns else master
            st.success("Edits saved ✔")

    with c2:
        if st.button("↩️ Revert unsaved edits", use_container_width=True):
            st.session_state.pop("_edit_buffer", None)
            # drop any editor key so it remounts cleanly
            for k in list(st.session_state.keys()):
                if str(k).startswith("editor_view_df_"):
                    st.session_state.pop(k, None)
            st.rerun()

    # what the user is currently seeing (with live formulas)
    st.session_state["current_view_df"] = edited.copy()

else:
    # Read-only view with the same formula columns
    st.dataframe(edit_base, use_container_width=True, hide_index=True)
    st.session_state["current_view_df"] = edit_base.copy()



st.write("")  # spacing

export_df = st.session_state["current_view_df"]

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇️ Download CSV",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="subitems_view.csv",
        mime="text/csv",
        key="dl_csv_after_edit",
        use_container_width=True,
    )
with c2:
    st.download_button(
        "⬇️ Download Excel (.xlsx)",
        data=to_excel_bytes({"Subitems": export_df}),
        file_name="subitems_view.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_xlsx_after_edit",
        use_container_width=True,
    )
if st.button("Reset data to original"):
    st.session_state["master_df"] = final_df.copy()
    st.session_state.pop("editor_view_df", None)
    st.success("Data reset to original values.")
    st.rerun()

# ===================== SUMMARY (Excel-aligned) ============================
st.write("")
if st.button("📊 Create summary", use_container_width=True):
    st.subheader("Quote summary (Excel-aligned)")

    # ===== SUMMARY uses the VISIBLE ROWS only, de-duped by subitem_id =====
    # take what you actually see in the table, else fall back
    src = st.session_state.get("current_view_df")
    if not isinstance(src, pd.DataFrame) or src.empty:
        src = st.session_state.get("master_df", final_df)
    src = src.copy()

    # Use each row once (protect against accidental repeats)
    if "subitem_id" in src.columns:
        src = src.drop_duplicates(subset="subitem_id")

    # --- Tiny debug readout so we can verify we're summing ~14 rows, not 1,280 ---
    with st.expander("ℹ️ Summary debug (rows & quick totals)", expanded=False):
        st.write("Rows used in summary:", len(src))


        def _num(col):
            return src[col].map(as_number).fillna(0.0) if col in src.columns else None


        dbg_cols = ["Quantity", "List Price $", "Amount ($)", "Discount ($)", "CT Cost ($)"]
        for c in dbg_cols:
            s = _num(c)
            if s is not None:
                st.write(f"Sum of **{c}**:", f"{float(s.sum()):,.2f}")


    # ---------- matching Excel logic, but let Amount($) drive when present ----------
    def _need(title):
        if title in src.columns: return title
        return best_name_match_simple(title, list(map(str, src.columns)))


    def _num_series(name, default=0.0):
        if not name or name not in src.columns:
            return pd.Series([default] * len(src), index=src.index, dtype=float)
        return src[name].map(as_number).fillna(default)


    col_qty = _need("Quantity")
    col_list_unit = _need("List Price $")
    col_amount = _need("Amount ($)")
    col_disc_amt = _need("Discount ($)")
    col_disc_pct = _need("Discount (%)")
    col_cost_unit = _need("CT Cost ($)")

    qty = _num_series(col_qty, 1.0)  # if missing, treat as 1
    lp = _num_series(col_list_unit, 0.0)
    amt = _num_series(col_amount, 0.0)
    discA = _num_series(col_disc_amt, 0.0)
    discP = _num_series(col_disc_pct, 0.0)
    costU = _num_series(col_cost_unit, 0.0)
    # --- Choose how to compute "Total List Price ($)" ---
    # OFF  -> sum of the unit "List Price $" column (your current expectation)
    # ON   -> Excel 'Amount' logic (SUM(Amount) or SUM(Quantity×List Price))
    use_amount_logic = st.checkbox(
        "Use Amount logic (Quantity × List Price)",
        value=False,
        help="Turn ON to match Excel SUM(Amount). Turn OFF to sum the unit List Price column only."
    )

    if use_amount_logic:
        # Excel/Amount path
        total_list = float(amt.sum()) if col_amount else float((qty * lp).sum())
        # Discounts computed off the same base
        if not col_disc_amt:
            base_for_disc = amt if col_amount else (qty * lp)
            discA = base_for_disc * (discP / 100.0)
    else:
        # Unit-sum path (what you want: just add the List Price column)
        total_list = float(lp.sum())
        # If Discount ($) missing, derive from pct * unit price (no quantity)
        if not col_disc_amt:
            discA = lp * (discP / 100.0)

    # Cost totals
    total_cost = float((qty * costU).sum())

    # Item-level net and margins
    total_discount = float(discA.sum())
    net_after_item_disc = total_list - total_discount
    gm_on_list = (1.0 - (total_cost / total_list)) if total_list > 0 else 0.0
    gm_after_item = (1.0 - (total_cost / net_after_item_disc)) if net_after_item_disc > 0 else 0.0

    # Overall discount input (like Summary!B10)
    overall_pct = st.number_input(
        "Overall Discount (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0,
        help="Matches the green input on the Excel Summary sheet."
    ) / 100.0

    overall_discount_amt = total_list * overall_pct
    net_after_overall = net_after_item_disc - overall_discount_amt
    gm_after_overall = (1.0 - (total_cost / net_after_overall)) if net_after_overall > 0 else 0.0


    def _money(x):
        return f"{x:,.2f}"


    def _pct(x):
        return f"{x * 100:,.1f}%"


    item_metrics = [
        ("Total List Price ($)", _money(total_list)),
        ("Total Discount Amount ($)", _money(total_discount)),
        ("Net Price After Item Discounts ($)", _money(net_after_item_disc)),
        ("Total Cost ($)", _money(total_cost)),
        ("Gross Margin on List Price (%)", _pct(gm_on_list)),
        ("Gross Margin after Item Discounts (%)", _pct(gm_after_item)),
    ]
    overall_metrics = [
        ("Overall Discount (%)", _pct(overall_pct)),
        ("Overall Discount Amount ($)", _money(overall_discount_amt)),
        ("Net Revenue After Overall Discount ($)", _money(net_after_overall)),
        ("Gross Margin after Overall Discount (%)", _pct(gm_after_overall)),
    ]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Item-Level Metrics**")
        st.table(pd.DataFrame(item_metrics, columns=["Metric", "Value"]))
    with c2:
        st.markdown("**Overall Discount Analysis**")
        st.table(pd.DataFrame(overall_metrics, columns=["Metric", "Value"]))
    # ===== END SUMMARY =====

