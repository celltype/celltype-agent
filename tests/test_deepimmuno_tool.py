def test_deepimmuno_tool_is_registered():
    from ct.tools import registry, ensure_loaded

    ensure_loaded()
    tool = registry.get_tool("protein.deepimmuno")

    assert tool is not None
    assert tool.name == "protein.deepimmuno"
    assert tool.category == "protein"
    assert tool.requires_gpu is False
    assert tool.cpu_only is True
    assert tool.min_vram_gb == 0
    assert tool.min_ram_gb == 8
    assert tool.docker_image == "celltype/deepimmuno:latest"
