# In this lab, you will step into the role of a technical AI lead at a fashion brand preparing a summer sunglasses campaign. Your task is to design a fully automated creative pipeline that mirrors a real-world business scenario. Instead of handling each piece manually, you will guide a system that scans online sources for emerging fashion trends, matches those trends with sunglasses in the internal catalog, designs a campaign visual, generates a short marketing quote, and finally packages everything into an executive-ready report.

# The goal is to experience how multiple agents, tools, and models can be orchestrated into a single, coherent workflow. By the end of this lab, you will have built a pipeline that feels less like a script of isolated steps and more like a small team working together to solve a creative challenge. 

# By completing this lab, you will see how to move beyond single-turn interactions with a model and instead design multi-agent pipelines that coordinate planning, research, and creative generation. You will learn how to ground agent reasoning in external tools so that outputs are not just imaginative but also supported by real data. You will also experiment with reflection and packaging steps that enforce quality control and prepare results for an executive audience.


# =========================
# Imports
# =========================

# --- Standard library ---
import base64
import json
import os
import re
from datetime import datetime
from io import BytesIO

# --- Third-party ---
import requests
import openai
from PIL import Image
from dotenv import load_dotenv
from IPython.display import Markdown, display
import aisuite

# --- Local / project ---
import tools
import utils


# =========================
# Environment & Client
# =========================
load_dotenv()
client = aisuite.Client()


# Agentic pipelines only become effective when the model is given explicit capabilities beyond its base reasoning. Declaring these tools upfront makes the agent’s action space unambiguous, ensures that prompts naturally guide tool selection, and keeps orchestration and testing transparent through well-defined interfaces.

# You will assemble a marketing research team, a group of specialized agents collaborating to design a summer sunglasses campaign. To empower them, we start by defining the tools that will ground their reasoning in real data.

# The first tool is tools.tavily_search_tool, which performs live web searches to uncover evidence of current fashion trends. Try it now by running a simple query for “trends in sunglasses fashion”:

tools.tavily_search_tool('trends in sunglasses fashion')

# The second tool is tools.product_catalog_tool, which returns the internal sunglasses catalog. Each entry includes details such as product name, ID, description, stock quantity, and price. This structured data will allow the agents to connect online fashion trends with actual items in stock:

tools.product_catalog_tool()

# Market Research Agent

# With the Market Research Agent, you take the first step in building your campaign. You ask it to scan the web with tavily_search_tool and uncover what’s trending in sunglasses fashion right now. Then you direct it to cross-check those signals against your internal catalog using product_catalog_tool, so you know which of your products fit the moment.

# The agent hands back a concise brief: the top fashion insights it found, the products that align with them, and a short explanation of why those picks make sense for your summer push. This gives you a clear, data-driven foundation to shape the rest of the campaign. 

def market_research_agent(return_messages: bool = False):

    utils.log_agent_title_html("Market Research Agent", "🕵️‍♂️")

    prompt_ = f"""
You are a fashion market research agent tasked with preparing a trend analysis for a summer sunglasses campaign.

Your goal:
1. Explore current fashion trends related to sunglasses using web search.
2. Review the internal product catalog to identify items that align with those trends.
3. Recommend one or more products from the catalog that best match emerging trends.
4. If needed, today date is {datetime.now().strftime("%Y-%m-%d")}.

You can call the following tools:
- tavily_search_tool: to discover external web trends.
- product_catalog_tool: to inspect the internal sunglasses catalog.

Once your analysis is complete, summarize:
- The top 2–3 trends you found.
- The product(s) from the catalog that fit these trends.
- A justification of why they are a good fit for the summer campaign.
"""
    messages = [{"role": "user", "content": prompt_}]
    tools_ = tools.get_available_tools()

    while True:
        response = client.chat.completions.create(
            model="openai:o4-mini",
            messages=messages,
            tools=tools_,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        if msg.content:
            utils.log_final_summary_html(msg.content)
            return (msg.content, messages) if return_messages else msg.content

        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                utils.log_tool_call_html(tool_call.function.name, tool_call.function.arguments)
                result = tools.handle_tool_call(tool_call)
                utils.log_tool_result_html(result)

                messages.append(msg)
                messages.append(tools.create_tool_response_message(tool_call, result))
        else:
            utils.log_unexpected_html()
            return ("[⚠️ Unexpected: No tool_calls or content returned]", messages) if return_messages else "[⚠️ Unexpected: No tool_calls or content returned]"
        
# Let’s try to get some advice from the Market Research Agent about our summer sunglasses campaign.

market_research_result = market_research_agent()

# Next, we’ll turn this brief into a visual concept with the Graphic Designer Agent.

# Graphic Designer Agent

# With the Graphic Designer Agent, you move from analysis to creativity.
# You take the brief from your Market Research Agent and ask this one to translate it into a visual concept.
# Because aisuite does not yet support direct image generation (like gpt-image-1-mini), you guide the process in two stages:

#     First, the agent uses aisuite with an OpenAI text model (o4-mini) to craft a vivid prompt and a short, engaging caption.
#     Then, the prompt is sent to OpenAI’s gpt-image-1-mini API to generate the campaign image itself.

# The result gives you everything you need to move forward: the generated image (saved locally for reuse), the exact prompt that produced it (useful for iteration), and a polished caption for campaign storytelling. 

import base64

def graphic_designer_agent(trend_insights: str, caption_style: str = "short punchy", size: str = "1024x1024") -> dict:
    utils.log_agent_title_html("Graphic Designer Agent", "🎨")

    # Step 1: Generate prompt and caption using aisuite
    system_message = (
        "You are a visual marketing assistant. Based on the input trend insights, "
        "write a creative and visual prompt for an AI image generation model, and also a short caption."
    )
    user_prompt = f"""
Trend insights:
{trend_insights}
Please output:
1. A vivid, descriptive prompt to guide image generation.
2. A marketing caption in style: {caption_style}.
Respond in this format:
{{"prompt": "...", "caption": "..."}}
"""
    chat_response = client.chat.completions.create(
        model="openai:o4-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
    )
    content = chat_response.choices[0].message.content.strip()
    match = re.search(r'\{.*\}', content, re.DOTALL)
    parsed = json.loads(match.group(0)) if match else {"error": "No JSON returned", "raw": content}
    prompt = parsed["prompt"]
    caption = parsed["caption"]

    # Step 2: Generate image with gpt-image-1-mini (returns base64, not URL)
    openai_client = openai.OpenAI()
    image_response = openai_client.images.generate(
        model="gpt-image-1-mini",
        prompt=prompt,
        size=size,
        quality="medium",   # gpt-image-1 accepts: low | medium | high | auto
        n=1,
        # no response_format — gpt-image-1 always returns b64_json
    )

    b64 = image_response.data[0].b64_json
    img_bytes = base64.b64decode(b64)
    img = Image.open(BytesIO(img_bytes))

    image_path = "generated_image.png"
    img.save(image_path)

    utils.log_final_summary_html(f"""
        <h3>Generated Image and Caption</h3>
        <p><strong>Image Path:</strong> <code>{image_path}</code></p>
        <p><strong>Generated Image:</strong></p>
        <img src="{image_path}" alt="Generated Image" style="max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 8px; margin-top: 10px; margin-bottom: 10px;">
        <p><strong>Prompt:</strong> {prompt}</p>
    """)

    return {
        "image_path": image_path,
        "prompt": prompt,
        "caption": caption,
    }
    
    
# Now let’s run the graphic_designer_agent to generate a campaign image, using the trend insights provided by the Market Research Agent.

graphic_designer_agent_result = graphic_designer_agent(
    trend_insights=market_research_result,
)

# With a visual in hand, you’ll craft the campaign voice using the Copywriter Agent.
# Copywriter Agent

# Once the Market Research Agent and Graphic Designer Agent have done their work, you now turn to the Copywriter Agent. With both the campaign image and the trend summary in hand, you ask this agent to create the voice of your campaign.

# It takes the visual and the analysis together as multimodal input and crafts a short, elegant marketing quote that captures the essence of the message. Along with the quote, it gives you a clear justification—why the phrase fits the image and how it ties back to the trends.

# This way, you don’t just get a catchy line, you also get the reasoning behind it, making it easier to defend and refine in front of stakeholders.

​def copywriter_agent(image_path: str, trend_summary: str, model: str = "openai:o4-mini") -> dict:

    """
    Uses aisuite (OpenAI only) to send an image and a trend summary and return a campaign quote.

    Args:
        image_path (str): URL of the image to be analyzed.
        trend_summary (str): Text from the researcher agent.
        model (str): OpenAI model (e.g., openai:o4-mini, openai:gpt-4o)

    Returns:
        dict: {
            "quote": "...",
            "justification": "...",
            "image_path": "..."
        }
    """

    utils.log_agent_title_html("Copywriter Agent", "✍️")

    # Step 1: Load local image and encode as base64
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    b64_img = base64.b64encode(img_bytes).decode("utf-8")

    # Step 2: Build OpenAI-compliant multimodal message
    messages = [
        {
            "role": "system",
            "content": "You are a copywriter that creates elegant campaign quotes based on an image and a marketing trend summary."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_img}",
                        "detail": "auto"
                    }
                },
                {
                    "type": "text",
                    "text": f"""
Here is a visual marketing image and a trend analysis:

Trend summary:
\"\"\"{trend_summary}\"\"\"

Please return a JSON object like:
{{
  "quote": "A short, elegant campaign phrase (max 12 words)",
  "justification": "Why this quote matches the image and trend"
}}"""
                }
            ]
        }
    ]

    # Step 3: Send request via aisuite
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    # Step 4: Parse JSON response
    content = response.choices[0].message.content.strip()

    utils.log_final_summary_html(content)

    try:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {"error": "No valid JSON returned"}
    except Exception as e:
        parsed = {"error": f"Failed to parse: {e}", "raw": content}


    parsed["image_path"] = image_path
    return parsed

# Next, let’s call the Copywriter Agent to generate a short campaign quote based on the marketing image and the trend insights produced earlier.

copywriter_agent_result = copywriter_agent(
    image_path=graphic_designer_agent_result["image_path"],
    trend_summary=market_research_result,
)

# With a quote and justification ready, you’ll package everything into an executive-ready artifact using the Packaging Agent.
# Packaging Agent

# Finally, you bring in the Packaging Agent to tie everything together. After the Market Research Agent, Graphic Designer Agent, and Copywriter Agent have each contributed their part, this agent compiles the entire story into one polished artifact.

# You ask it to take the trend summary, the campaign visual, the crafted quote, and the justification, and assemble them into an executive-ready markdown report. Along the way, it rewrites the trend insights for clarity and tone, ensures the quote is styled properly with the image, and organizes everything so the final document looks professional and persuasive.

# With this step, you end up with a complete campaign package—easy to share, visually engaging, and ready for CEO-level review.

def packaging_agent(trend_summary: str, image_url: str, quote: str, justification: str, output_path: str = "campaign_summary.md") -> str:

    """
    Packages the campaign assets into a beautifully formatted markdown report for executive review.

    Args:
        trend_summary (str): Summary of the market trends.
        image_url (str): URL of the campaign image.
        quote (str): Marketing quote to overlay.
        justification (str): Explanation for the quote.
        output_path (str): Path to save the markdown report.

    Returns:
        str: Path to the saved markdown file.
    """

    utils.log_agent_title_html("Packaging Agent", "📦")

    # We use this path in the src of the <img>
    styled_image_html = f"""
        ![Open the generated file to see]({image_url})          
    """

    beautified_summary = client.chat.completions.create(
        model="openai:o4-mini",
        messages=[{"role": "system", "content": "You are a marketing communication expert writing elegant campaign summaries for executives."},
            {"role": "user", "content": 
            f"""
                Please rewrite the following trend summary to be clear, professional, and engaging for a CEO audience:  \"\"\"{trend_summary.strip()}\"\"\"
            """}]
    ).choices[0].message.content.strip()

    utils.log_tool_result_html(beautified_summary)

    # Combine all parts into markdown
    markdown_content = f"""
        # 🕶️ Summer Sunglasses Campaign – Executive Summary

            ## 📊 Refined Trend Insights
            {beautified_summary}

            ## 🎯 Campaign Visual
            {styled_image_html}

            ## ✍️ Campaign Quote
            {quote.strip()}

            ## ✅ Why This Works
            {justification.strip()}

            ---

            *Report generated on {datetime.now().strftime('%Y-%m-%d')}*
        """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return output_path

packaging_agent_result = packaging_agent(
    trend_summary=market_research_result,
    image_url=graphic_designer_agent_result["image_path"],
    quote=copywriter_agent_result["quote"],
    justification=copywriter_agent_result["justification"],
    output_path=f"campaign_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
)

# Load and render the markdown content
with open(packaging_agent_result, "r", encoding="utf-8") as f:
    md_content = f.read()

display(Markdown(md_content))

# In this step, you will define a single function, run_sunglasses_campaign_pipeline, that ties all the pieces together into one seamless workflow for your summer sunglasses campaign.

# The function will:

#     Run market research to scan fashion trends and match them to your catalog.
#     Generate a visually styled image and caption.
#     Create a short, elegant campaign quote with justification.
#     Package everything into a polished markdown report tailored for executive review.

# By defining this function, you make it easy to run the entire pipeline in one call while still being able to trace intermediate results and view the final report. 

def run_sunglasses_campaign_pipeline(output_path: str = "campaign_summary.md") -> dict:
    """
    Runs the full summer sunglasses campaign pipeline:
    1. Market research (search trends + match products)
    2. Generate visual + caption
    3. Generate quote based on image + trend
    4. Create executive markdown report

    Returns:
        dict: Dictionary containing all intermediate results + path to final report
    """
    # 1. Run market research agent
    trend_summary = market_research_agent()
    print("✅ Market research completed")

    # 2. Generate image + caption
    visual_result = graphic_designer_agent(trend_insights=trend_summary)
    image_path = visual_result["image_path"]
    print("🖼️ Image generated")

    # 3. Generate quote based on image + trends
    quote_result = copywriter_agent(image_path=image_path, trend_summary=trend_summary)
    quote = quote_result.get("quote", "")
    justification = quote_result.get("justification", "")
    print("💬 Quote created")

    # 4. Generate markdown report
    md_path = packaging_agent(
        trend_summary=trend_summary,
        image_url=image_path,  
        quote=quote,
        justification=justification,
        output_path=f"campaign_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
    )

    print(f"📦 Report generated: {md_path}")

    return {
        "trend_summary": trend_summary,
        "visual": visual_result,
        "quote": quote_result,
        "markdown_path": md_path
    }

results = run_sunglasses_campaign_pipeline()

with open(results["markdown_path"], "r", encoding="utf-8") as f:
    md_content = f.read()
display(Markdown(md_content))

# By completing this lab, you have seen how to:

#     Use multi-agent LLM pipelines to automate a creative workflow end-to-end.
#     Combine reasoning, tool-calling, and external data to ground your outputs in reality.
#     Apply multimodal models (like gpt-4o) that process both text and images for tasks such as generating campaign quotes.
#     Extend the model’s abilities with tools (tavily_search_tool, product_catalog_tool) so your outputs are not only imaginative but also practical.
#     Keep execution transparent and debuggable with structured logging and HTML-styled blocks.
#     Deliver a polished, executive-ready report in Markdown format that blends insights, visuals, and justifications into a single artifact.
