import json
from typing import Any, MutableMapping

import dotenv
import streamlit as st

from ui.asyncbridge import AsyncBridge
from ui.logger import get_custom_logger
from ui.utils import BackendConnectionException, get_server_url, make_handshake, make_query

dotenv.load_dotenv()

logger = get_custom_logger(logger_name="hsr-frontend")

# === Helpers ===


def get_streamlit_session_id() -> str | None:
    """Return the current Streamlit browser session identifier when available."""
    try:
        session_id = getattr(st.context, "session_id", None)
        if session_id is None:
            return None
        return str(session_id)
    except Exception as e:
        logger.warning(f"Exception caught while getting st.context.session_id: {e}")
        return None


def cleanup_async_bridge(session_state: MutableMapping[str, Any]) -> None:
    """Close and remove any existing bridge associated with this session."""
    bridge = session_state.pop("async_bridge", None)
    session_state.pop("async_bridge_session_id", None)
    if bridge is not None:
        try:
            bridge.close()
        except Exception as e:
            logger.warning(f"Exception caught while cleaning up AsynBridge: {e}")
            pass


def ensure_async_bridge(
    session_state: MutableMapping[str, Any] | Any,
    *,
    session_id: str | None = None,
) -> AsyncBridge:
    """Create a single AsyncBridge per session at startup and after a browser refresh."""
    current_session_id = session_id if session_id is not None else get_streamlit_session_id()
    existing_bridge = session_state.get("async_bridge")
    existing_session_id = session_state.get("async_bridge_session_id")

    if isinstance(existing_bridge, AsyncBridge):
        if current_session_id is None or existing_session_id == current_session_id:
            return existing_bridge
        cleanup_async_bridge(session_state)

    bridge = AsyncBridge()
    bridge.start()
    session_state["async_bridge"] = bridge
    session_state["async_bridge_session_id"] = current_session_id
    return bridge


def do_search(
    bridge: AsyncBridge,
    query: str,
    top_k: int,
    vector_search_weight: float = 0.5,
    rrf_k: int = 60,
    hybrid_search_fold: int = 3,
) -> tuple[list[str], bool]:

    data = {
        "query": query,
        "top_k": top_k,
        "vector_search_weight": vector_search_weight,
        "rrf_k": rrf_k,
        "hybrid_search_fold": hybrid_search_fold,
    }
    server_url = get_server_url()
    server_endpoint = f"{server_url}/v1/search/"

    search_results = []
    success = False
    with st.spinner("Searching ..."):
        try:
            response = make_query(server_endpoint, payload=data, bridge=bridge)
            if response.status_code == 200:
                resp = response.json()
                if "summary" in resp:
                    search_results.append(resp["summary"])
                if "hits" in resp:
                    search_results.extend(resp["hits"])
                success = True
            else:
                error_msg = f"HTTP status_code: {response.status_code}: {response.reason}\n"
                error_msg += json.dumps(response.json())
                search_results.append(error_msg)
        except BackendConnectionException as exc:  # pragma: no cover - defensive UI fallback
            msg = f"Unrecoverable failure to connect to HSR backend at {server_endpoint}: {exc.message}"
            search_results.append(msg)

        return search_results, success


def parse_result(result: str) -> tuple[str, str]:
    """Extract Source header and result body"""

    start_src_tag = "[Source: "  # len(start_src_tag) is 9
    pos_left = result.find(start_src_tag)
    if pos_left == -1:
        return "", ""
    end_src_tag = " | Type:"
    pos_right = result.find(end_src_tag)
    if pos_right == -1:
        return "", ""

    source = result[pos_left + len(start_src_tag) : pos_right]

    start_body_tag = "]\n\n"
    pos_left = result.find(start_body_tag)
    if pos_left == -1:
        return source, ""
    body = result[pos_left + len(start_body_tag) :]

    return source, body.strip()


def render_results(results: list[str], top_k: int, rrf_fold: int) -> None:
    """Render results in a scrollable, expandable list."""
    if not results:
        st.info("_Enter a query in the field at the bottom:_ (click on the field first)")
        return

    st.subheader(f"Top {top_k} ranked results:")
    st.caption(results[0])

    st.markdown(
        "<div style='max-height: 420px; overflow-y: auto; padding-right: 0.25rem;'>",
        unsafe_allow_html=True,
    )
    if len(results) > 1:
        for k, hit in enumerate(results[1:], 1):
            if k > top_k:
                break
            source, body = parse_result(hit)
            with st.expander(f"{k}- {source}", expanded=False):
                st.markdown(body)
    st.markdown("</div>", unsafe_allow_html=True)


# === UI ===

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        Given a corpus of indexed local documents (of supported types) the parmeters below
        affect the RRF based ranking of the document chunks retrieved against a user _query_
        """
    )

    st.markdown(
        """
        ## Search parameters setting:

        ### _Keyword_ search
        - Set the _Vector Search weight_ (slidebar) to **0.0**
        - Set `top_k` (maximum number of hits to select)
        - RRF parameters are not applicable (and ignored)

        ### _Vector_ search
        - Set the _Vector Search weight_ (slidebar) to **1.0**
        - Set `top_k` (maximum number of hits to select)
        - RRF parameters are not pplicable (and ignored)

        ### _Hybrid_ search (a mix of _Keyword_ and _Vector_ search)
        - Set `top_k` (maximum number of hits to select)
        - Set the _Vector Search weight_ (slidebar) to a value **strictly > 0.0 and < 1.0**
        - RRF parameters
            - `rrf_fold`: multiplier for carrying out an extended _Keyword_ and _Vector_ (generate `rrf_fold * top_k` results).
               However, in some boundary cases, the actual hits can less then `rrf_fold * top_k`
            - `K`: `K` parameter in the RRF canonical formula (defaults to 60, the industry standard)

        """
    )

    st.write("---")

    top_k = st.number_input("top_k: Maximum number of top-ranking hits to select", min_value=1, value=5)

    weight = st.slider("Vector Search weight", 0.0, 1.0, 0.5, format="plain", label_visibility="visible")

    rrf_k = st.number_input(
        "K parameter in Reciprocal Rank Fusion (RRF)",
        min_value=0,
        value=60,
        disabled=False if (weight > 0.0 and weight < 1.0) else True,
    )

    rrf_fold = st.number_input(
        "Hybrid Search Fold: The first top_k * rrf_fold results are selected in general Hybrid Search before RRF"
        " (this parameter is ignored in pure Keyword or Semantic search)",
        min_value=1,
        value=3,
        disabled=False if (weight > 0.0 and weight < 1.0) else True,
    )

    st.caption(f"Top-k set to {top_k}; RRF k={rrf_k}; fold={rrf_fold}")

    st.session_state.top_k = top_k
    st.session_state.weight = weight
    st.session_state.rrf_k = rrf_k
    st.session_state.rrf_fold = rrf_fold

st.title("HSR: _Hybrid Search Ranker_")
st.info(
    """
    Set the desired **search type** on the left-hand column:
    - **_Keyword_**
    - **_Semantic_**
    - **_Hybrid_** (_Keyword/Vector_ blend) with **_Reciprocal Rank Fusion_** (RRF)

    Test the results on a pre-indexed corpus of local data
    """
)

# Wrap the equation and explanation inside a collapsible expander
with st.expander("ℹ️ Show/Hide Reciprocal Rank Fusion (RRF) Equation", expanded=True):
    st.latex(r"""
    RRF\_Score(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}
    """)

    st.markdown("""
    **Where:**
    * $D$ is the set of all documents.
    * $M$ is the set of retrieval models (e.g., lexical and semantic).
    * $r_m(d)$ is the rank of document $d$ in retrieval system $m$.
    * $k$ is a constant hyperparameter (commonly set to $60$).
    """)

if "async_bridge" not in st.session_state or not st.session_state["async_bridge"].is_valid():
    ensure_async_bridge(st.session_state, session_id=get_streamlit_session_id())

# Expose the live bridge instance to the rest of the streamlit module for future development.
bridge_value = st.session_state.get("async_bridge")
if not isinstance(bridge_value, AsyncBridge):
    raise RuntimeError("AsyncBridge is missing from the current Streamlit session state.")
async_bridge = bridge_value

if "handshake" not in st.session_state:  # UI startup or browser refresh
    st.session_state.handshake = True

    with st.spinner("Contacting HSR backend - _please wait..._"):
        error_msg = "HSR backend unresponsive, please retry later by refreshing the browser session. "
        error_msg += f"If the problem persists, troubleshoot the backend {get_server_url()}"
        try:
            response = make_handshake(async_bridge)
            if response.status_code == 200:
                st.success("HSR backend is alive, please enter query")
            else:
                st.error(error_msg)
        except BackendConnectionException:
            st.error(error_msg)

if "results" not in st.session_state:
    st.session_state.results = {}

query = st.chat_input("Enter query here ...")
ok = True
if query:
    st.session_state.results, ok = do_search(
        async_bridge,
        query,
        st.session_state.top_k,
        vector_search_weight=st.session_state.weight,
        rrf_k=st.session_state.rrf_k,
        hybrid_search_fold=st.session_state.rrf_fold,
    )
    st.session_state.last_query = query

if "last_query" in st.session_state and st.session_state.last_query:
    if ok:
        st.info(f"Query results for: {st.session_state.last_query}")

        render_results(
            st.session_state.results,
            st.session_state.top_k,
            st.session_state.rrf_fold,
        )
    else:
        st.info(f"ERROR: {st.session_state.results} - see logger for further details.")
