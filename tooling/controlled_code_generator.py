from __future__ import annotations

from tooling.policy_layer import PolicyLayer
from tooling.tool_spec import GeneratedTool, ToolSpec, ToolValidationReport
from tooling.tool_templates import TEMPLATE_RENDERERS


class ControlledCodeGenerator:
    """Whitelisted code generator.

    v3.119 keeps generation deterministic and safe by emitting code only from approved
    templates. The only tunable surface is a small set of validated template parameters.
    """

    CAPABILITY_TO_TEMPLATE = {
        "text_summarizer": "text_summarizer",
        "keyword_extractor": "keyword_extractor",
        "numeric_stats": "numeric_stats",
    }

    def __init__(self, policy_layer: PolicyLayer | None = None) -> None:
        self.policy_layer = policy_layer or PolicyLayer()

    def supported_capabilities(self) -> set[str]:
        return set(self.CAPABILITY_TO_TEMPLATE)

    def make_spec(
        self,
        capability: str,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, object] | None = None,
    ) -> ToolSpec:
        if capability not in self.CAPABILITY_TO_TEMPLATE:
            raise ValueError(f"Unsupported capability for controlled generation: {capability}")
        template_name = self.CAPABILITY_TO_TEMPLATE[capability]
        default_name = f"generated_{capability}"
        default_description = f"Safely generated tool for capability {capability}."
        merged_parameters: dict[str, object] = {"max_sentences": 3} if capability == "text_summarizer" else {}
        if parameters:
            merged_parameters.update(parameters)
        return ToolSpec(
            name=name or default_name,
            capability=capability,
            description=description or default_description,
            template_name=template_name,
            parameters=merged_parameters,
        )

    def generate(self, spec: ToolSpec) -> GeneratedTool:
        renderer = TEMPLATE_RENDERERS.get(spec.template_name)
        if renderer is None:
            raise ValueError(f"Unsupported template: {spec.template_name}")
        source_code = renderer(spec)
        validation = self.policy_layer.validate_source(source_code)
        return GeneratedTool(spec=spec, source_code=source_code, validation=validation)

    def validate(self, source_code: str) -> ToolValidationReport:
        return self.policy_layer.validate_source(source_code)
