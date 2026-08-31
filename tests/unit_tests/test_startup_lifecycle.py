import sys
import threading


def test_ui_module_import_is_side_effect_free() -> None:
    sys.modules.pop("ui.main", None)

    before = {thread.name for thread in threading.enumerate()}
    import ui.main  # noqa: F401

    after = {thread.name for thread in threading.enumerate()}

    extra = after - before
    assert not any(name.startswith("ThreadPoolExecutor") or name.startswith("asyncio_") for name in extra)
