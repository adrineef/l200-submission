import logging
import os

from google import genai
from google.genai import types as genai_types
from google.adk.tools import ToolContext

logger = logging.getLogger("haiku_agent")


async def generate_haiku_illustration(prompt: str, tool_context: ToolContext) -> dict:
    """Génère une illustration japonaise (ukiyo-e) pour un haïku.

    Args:
        prompt: Description visuelle de la scène du haïku, en une ou deux phrases.

    Returns:
        status="success" et le nom de l'artefact, ou status="error" avec un recovery_hint.
    """
    try:
        client = genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ.get("IMAGE_LOCATION", "global"),
        )
        styled = f"Japanese ukiyo-e / sumi-e illustration, soft colors, no text. {prompt}"
        res = client.models.generate_content(
            model=os.environ.get("IMAGE_MODEL", "gemini-2.5-flash-image"),
            contents=styled,
        )
        for part in res.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                await tool_context.save_artifact(
                    "haiku_illustration.png",
                    genai_types.Part.from_bytes(
                        data=part.inline_data.data,
                        mime_type=part.inline_data.mime_type or "image/png",
                    ),
                )
                return {"status": "success", "artifact": "haiku_illustration.png"}
        return {"status": "error", "message": "Aucune image dans la réponse.",
                "recovery_hint": "Reformule le prompt de façon plus neutre et réessaie une fois."}
    except Exception as e:
        logger.exception("illustration_failed")
        return {"status": "error", "message": type(e).__name__,
                "recovery_hint": "Continue sans illustration et préviens l'utilisateur."}
