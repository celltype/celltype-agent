"""Tests for tool registry GPU metadata support."""

import pytest


class TestGPUToolRegistration:
    """Test GPU metadata in tool registration."""

    def test_register_gpu_tool(self):
        from ct.tools import ToolRegistry

        reg = ToolRegistry()

        @reg.register(
            name="test.gpu_tool",
            description="A test GPU tool",
            category="test",
            parameters={"input": "test input"},
            requires_gpu=True,
            gpu_profile="structure",
            estimated_cost=0.10,
            docker_image="celltype/test:latest",
        )
        def test_gpu_tool(input: str = "", **kwargs):
            return {"summary": "test"}

        tool = reg.get_tool("test.gpu_tool")
        assert tool is not None
        assert tool.requires_gpu is True
        assert tool.gpu_profile == "structure"
        assert tool.estimated_cost == 0.10
        assert tool.docker_image == "celltype/test:latest"

    def test_register_non_gpu_tool_defaults(self):
        from ct.tools import ToolRegistry

        reg = ToolRegistry()

        @reg.register(
            name="test.normal_tool",
            description="A normal tool",
            category="test",
        )
        def test_normal_tool(**kwargs):
            return {"summary": "test"}

        tool = reg.get_tool("test.normal_tool")
        assert tool is not None
        assert tool.requires_gpu is False
        assert tool.gpu_profile == ""
        assert tool.estimated_cost == 0.0
        assert tool.docker_image == ""
        assert tool.min_ram_gb == 0
        assert tool.cpu_only is False
        assert tool.num_gpus == 1

    def test_register_cpu_only_tool(self):
        from ct.tools import ToolRegistry

        reg = ToolRegistry()

        @reg.register(
            name="test.cpu_tool",
            description="A CPU-only high-memory tool",
            category="test",
            cpu_only=True,
            min_ram_gb=64,
            num_gpus=0,
        )
        def test_cpu_tool(**kwargs):
            return {"summary": "test"}

        tool = reg.get_tool("test.cpu_tool")
        assert tool.cpu_only is True
        assert tool.min_ram_gb == 64
        assert tool.num_gpus == 0

    def test_register_multi_gpu_tool(self):
        from ct.tools import ToolRegistry

        reg = ToolRegistry()

        @reg.register(
            name="test.multi_gpu",
            description="A multi-GPU tool",
            category="test",
            requires_gpu=True,
            num_gpus=2,
            min_vram_gb=80,
        )
        def test_multi_gpu(**kwargs):
            return {"summary": "test"}

        tool = reg.get_tool("test.multi_gpu")
        assert tool.num_gpus == 2
        assert tool.min_vram_gb == 80


class TestGPUToolListing:
    """Test GPU status in tool listing."""

    def test_list_tools_table_shows_gpu(self):
        from ct.tools import ToolRegistry

        reg = ToolRegistry()

        @reg.register(
            name="test.gpu_tool",
            description="GPU tool",
            category="test",
            requires_gpu=True,
            estimated_cost=0.10,
        )
        def gpu_tool(**kwargs):
            return {"summary": "test"}

        table = reg.list_tools_table()
        # Table should be renderable
        assert table is not None
        assert table.title == "ct Tools"

    def test_tool_descriptions_include_gpu_info(self):
        from ct.tools import ToolRegistry

        reg = ToolRegistry()

        @reg.register(
            name="test.gpu_tool",
            description="GPU tool",
            category="test",
            parameters={"seq": "amino acid sequence"},
            requires_gpu=True,
            estimated_cost=0.10,
        )
        def gpu_tool(**kwargs):
            return {"summary": "test"}

        desc = reg.tool_descriptions_for_llm()
        assert "GPU" in desc
        assert "~$0.10/run" in desc

    def test_tool_descriptions_include_cpu_only_info(self):
        from ct.tools import ToolRegistry

        reg = ToolRegistry()

        @reg.register(
            name="test.cpu_tool",
            description="CPU-only high-memory tool",
            category="test",
            parameters={"seq": "input"},
            cpu_only=True,
            min_ram_gb=64,
            estimated_cost=0.05,
        )
        def cpu_tool(**kwargs):
            return {"summary": "test"}

        desc = reg.tool_descriptions_for_llm()
        assert "CPU" in desc
        assert "64GB" in desc

    def test_list_tools_table_cpu_only_status(self):
        from ct.tools import ToolRegistry

        reg = ToolRegistry()

        @reg.register(
            name="test.cpu_tool",
            description="CPU tool",
            category="test",
            cpu_only=True,
            min_ram_gb=64,
            estimated_cost=0.05,
        )
        def cpu_tool(**kwargs):
            return {"summary": "test"}

        table = reg.list_tools_table()
        assert table is not None


class TestGPUToolsLoaded:
    """Test that GPU structure tools are loadable."""

    def test_gpu_structure_tools_registered(self):
        from ct.tools import registry, ensure_loaded

        ensure_loaded()

        esmfold = registry.get_tool("structure.esmfold")
        assert esmfold is not None
        assert esmfold.requires_gpu is True
        assert esmfold.gpu_profile == "structure"

        diffdock = registry.get_tool("structure.diffdock")
        assert diffdock is not None
        assert diffdock.requires_gpu is True
        assert diffdock.gpu_profile == "docking"

        # AlphaFold3 not yet available
        assert registry.get_tool("structure.alphafold3") is None

    def test_gpu_design_tools_registered(self):
        from ct.tools import registry, ensure_loaded

        ensure_loaded()

        rfdiffusion = registry.get_tool("design.rfdiffusion")
        assert rfdiffusion is not None
        assert rfdiffusion.requires_gpu is True
        assert rfdiffusion.category == "design"

        proteinmpnn = registry.get_tool("design.proteinmpnn")
        assert proteinmpnn is not None
        assert proteinmpnn.requires_gpu is True
        assert proteinmpnn.category == "design"

        bindcraft = registry.get_tool("design.bindcraft")
        assert bindcraft is not None
        assert bindcraft.requires_gpu is True
        assert bindcraft.gpu_profile == "structure"
        assert bindcraft.category == "design"
        assert bindcraft.docker_image == "celltype/bindcraft:latest"

    def test_gpu_tools_have_vram_requirements(self):
        from ct.tools import registry, ensure_loaded

        ensure_loaded()

        esmfold = registry.get_tool("structure.esmfold")
        assert esmfold.min_vram_gb == 32

        diffdock = registry.get_tool("structure.diffdock")
        assert diffdock.min_vram_gb == 32

        bindcraft = registry.get_tool("design.bindcraft")
        assert bindcraft.min_vram_gb == 32
