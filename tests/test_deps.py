import os


def test_mcp_sdk_importable_in_ci():
    # This is the exact failure we saw on RunPod: `mcp.server.fastmcp` missing.
    # In CI we install requirements.txt, so treat this as a hard requirement there.
    if os.getenv("CI", "").lower() != "true":
        return

    import mcp.server.fastmcp  # noqa: F401
    import mcp.server.sse  # noqa: F401

