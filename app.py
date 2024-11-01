import streamlit as st
import os
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai import APIClient, Credentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API key and project ID
watsonx_api_key = "djaQbKrnM6kHzAYKr1ftcB9IvJWnZ0pmN-9A7Px2twLT"
project_id = "69a51a7e-d0bf-41e6-a35e-ec661b5ae858"
wx_url = "https://us-south.ml.cloud.ibm.com"

# Check if API Key and Project ID exist
if not all([watsonx_api_key, project_id, wx_url]):
    st.error("Please set the API_KEY, PROJECT_ID, and WX_URL in the environment variables.")
    st.stop()

# Initialize API Client
credentials = Credentials(url=wx_url, api_key=watsonx_api_key)
api_client = APIClient(credentials, project_id=project_id)

# Task Options
task_options = {
    "qa": "問答",
    "text-generation": "文本生成",
    "summarization": "摘要",
    "translation": "翻譯",
    "code-generation": "程式碼生成"
}

# Model Options
model_options = {
    "meta-llama/llama-3-1-70b-instruct": "Llama 3-1 70B Instruct",
    "mistralai/mistral-large": "Mistral Large",
    "meta-llama/llama-3-1-8b-instruct": "LLAMA 3-1 8B INSTRUCT",
    "ibm/granite-3-8b-instruct": "Granite 3 8B Instruct",
    "ibm/granite-3-2b-instruct": "Granite 3 2B Instruct",
    "meta-llama/llama-3-405b-instruct": "Llama 3 405B Instruct",
}

# Task-specific prompt templates
task_prompts = {
    "qa":(
        "請用台灣繁體中文回覆所有內容。"
        "請根據使用者的問題進行回答"
        "輸入的問題是：\n{input_text}\n"
        "請注意生成的文本應該流暢且自然，避免重複和不相關的訊息。"
        "不清楚的問題，清不要亂回答"
    ),
    "text-generation": (
        "請用台灣繁體中文回覆所有內容。"
        "請根據以下提供的文本，創建一段連貫且有邏輯的後續內容。"
        "請確保生成的內容符合上下文，並保持語氣和風格的一致。"
        "輸入的文本是：\n{input_text}\n"
        "請注意生成的文本應該流暢且自然，避免重複和不相關的訊息。"
    ),
    "summarization": (
        "請用台灣繁體中文回覆所有內容。"
        "請閱讀以下文本，並根據其內容生成一個簡明扼要的摘要。"
        "摘要應該涵蓋主要的觀點和重要資訊，並保持原意。"
        "輸入的文本是：\n{input_text}\n"
        "請確保摘要簡潔明了，不超過指定字數，且不包含不相關的細節。"
        "請注意生成的文本應該流暢且自然，避免重複和不相關的訊息。"
    ),
    "code-generation": (
        "你是一個寫程式的高手，請根據以下描述，編寫相應的程式碼。"
        "程式碼應該具有良好的結構和註釋，並按照最佳實踐進行編寫。"
        "請確保程式碼能夠執行並達到預期功能。"
        "描述如下：\n{input_text}\n"
        "請注意，輸出的程式碼應該精簡明確，並包含必要的註釋來解釋其邏輯。"
        "程式碼生成完成後就請停止生成內容，不要生成不必要的內容。"
        "請注意生成的文本應該流暢且自然，避免重複和不相關的訊息。"
    ),
    "translation": (
        "你是一個翻譯高手，請將以下提供的文本翻譯為流暢的台灣繁體中文。"
        "請確保翻譯保留原始文本的含義和語氣，並且符合繁體中文的語法規範。"
        "輸入的文本是：\n{input_text}\n"
        "請注意，翻譯應避免逐字翻譯，應該根據上下文來進行流暢自然的轉換。"
        "請注意生成的文本應該流暢且自然，避免重複和不相關的訊息。"
        "翻譯完成後就請停止生成內容，不要生成不必要的內容。"
    ),
}

# Default Parameters
default_parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 500,
    "min_new_tokens": 30,
    "stop_sequences": []
    # "stop_sequences": "\n\n"
}

# Custom Styles
st.markdown("""
    <style>
        .title { font-size: 35px; color: #4B8BBE; text-align: center; font-weight: bold; margin-bottom: 20px; }
        .header { font-size: 20px; color: #306998; font-weight: bold; margin-top: 20px; }
        .metric { font-size: 18px; color: #FF6347; font-weight: bold; margin-top: 10px; padding: 10px; background-color: #f4f4f4; border-radius: 10px; text-align: center; border: 1px solid #ddd; }
        .upload-box { background-color: #f0f0f0; padding: 20px; border-radius: 10px; margin-top: 20px; }
        .container { max-width: 1800px; margin: 0 auto; }
        .output-box { border: 2px solid #4B8BBE; padding: 15px; border-radius: 10px; background-color: #f9f9f9; margin-top: 20px; color: #000000; }
        .model-header { background-color: #4B8BBE; color: #ffffff; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI Layout
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<div class="title">LLM Model Comparison Arena</div>', unsafe_allow_html=True)
st.sidebar.title("Model & Task Selection")


# Task selection
task_type = st.sidebar.selectbox("Select Task", list(task_options.keys()), format_func=lambda x: task_options[x])
st.sidebar.markdown(f"**Task Type:** {task_options[task_type]}")


# Model 1 selection
model_1_id = "ibm/granite-3-8b-instruct"

# Parameters for Model 1
st.sidebar.markdown(f"### {model_options[model_1_id]} Parameters")
decoding_method_1 = st.sidebar.radio(f"Select Decoding Method for {model_options[model_1_id]}", ["greedy","sample"], key="decoding_method_1")
min_tokens_1 = st.sidebar.number_input(f"Min Tokens for {model_options[model_1_id]}", min_value=1, max_value=500, value=1, key="min_tokens_1")
max_tokens_1 = st.sidebar.number_input(f"Max Tokens for {model_options[model_1_id]}", min_value=10, max_value=2000, value=500, key="max_tokens_1")
temperature_1 = st.sidebar.slider(f"Temperature for {model_options[model_1_id]}", min_value=0.0, max_value=1.0, value=0.7, key="temperature_1") if decoding_method_1 == "sample" else None
top_k_1 = st.sidebar.slider(f"Top-K for {model_options[model_1_id]}", min_value=1, max_value=100, value=50, key="top_k_1") if decoding_method_1 == "sample" else None
top_p_1 = st.sidebar.slider(f"Top-P for {model_options[model_1_id]}", min_value=0.0, max_value=1.0, value=1.0, key="top_p_1") if decoding_method_1 == "sample" else None


# Model 2 selection
model_2_id = st.sidebar.selectbox("Select Model 2", list(model_options.keys()), format_func=lambda x: model_options[x])

# Parameters for Model 2
# st.sidebar.markdown(f"### {model_options[model_2_id]} Parameters")
# decoding_method_2 = st.sidebar.radio(f"Select Decoding Method for {model_options[model_2_id]}", ["sample", "greedy"], key="decoding_method_2")
# min_tokens_2 = st.sidebar.number_input(f"Min Tokens for {model_options[model_2_id]}", min_value=1, max_value=500, value=1, key="min_tokens_2")
# max_tokens_2 = st.sidebar.number_input(f"Max Tokens for {model_options[model_2_id]}", min_value=10, max_value=2000, value=1000, key="max_tokens_2")
# temperature_2 = st.sidebar.slider(f"Temperature for {model_options[model_2_id]}", min_value=0.0, max_value=1.0, value=0.7, key="temperature_2") if decoding_method_2 == "sample" else None
# top_k_2 = st.sidebar.slider(f"Top-K for {model_options[model_2_id]}", min_value=1, max_value=100, value=50, key="top_k_2") if decoding_method_2 == "sample" else None
# top_p_2 = st.sidebar.slider(f"Top-P for {model_options[model_2_id]}", min_value=0.0, max_value=1.0, value=1.0, key="top_p_2") if decoding_method_2 == "sample" else None

# User Input Text
input_text = st.text_area("Input Text", placeholder="Enter the text you want to process...", height=150)

# Submit Button
if st.button("Submit Task"):
    # Create model instances with user-defined parameters
    parameters_1 = {
        "decoding_method": decoding_method_1,
        "max_new_tokens": max_tokens_1,
        "min_new_tokens": min_tokens_1,
        "repetition_penalty": 1.5,
        "stop_sequences": ["\n\n"]
        # "stop_sequences": "\n\n"
        # "temperature": temperature_1,
        # "top_k": top_k_1,
        # "top_p": top_p_1,
    }

    parameters_2 = {
        "decoding_method": decoding_method_1,
        "max_new_tokens": max_tokens_1,
        "min_new_tokens": min_tokens_1,
        "repetition_penalty": 1.5,
        "stop_sequences": ["\n\n"]
        # "temperature": temperature_1,
        # "top_k": top_k_1,
        # "top_p": top_p_1,
    }

    model_1 = WatsonxLLM(
        model_id=model_1_id,
        watsonx_client=api_client,
        params=parameters_1
    )

    model_2 = WatsonxLLM(
        model_id=model_2_id,
        watsonx_client=api_client,
        params=parameters_2
    )

    model_3 = WatsonxLLM(
        model_id="meta-llama/llama-3-405b-instruct",  # Fixed evaluation model
        watsonx_client=api_client,
        params=default_parameters
    )

    # Generate prompt for selected task type
    input_prompt = task_prompts[task_type].format(input_text=input_text)
    st.warning(input_prompt)
    prompt_list = [input_prompt]  # Wrap input text as list

    # Model 1 and Model 2 outputs
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="model-header">{model_options[model_1_id]}</div>', unsafe_allow_html=True)
        model_1_output_placeholder = st.empty()  # Create placeholder for streaming output
        model_1_text = ""
        for chunk in model_1.stream(input=prompt_list):
            model_1_text += chunk  # Accumulate chunk text
            model_1_output_placeholder.markdown(f'<div class="output-box">{model_1_text}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="model-header">{model_options[model_2_id]}</div>', unsafe_allow_html=True)
        model_2_output_placeholder = st.empty()  # Create placeholder for streaming output
        model_2_text = ""
        for chunk in model_2.stream(input=prompt_list):
            model_2_text += chunk  # Accumulate chunk text
            model_2_output_placeholder.markdown(f'<div class="output-box">{model_2_text}</div>', unsafe_allow_html=True)

    # Evaluation by Model 3
    st.markdown('<div class="model-header">Evaluation by Model 3 (Llama 3 405B Instruct)</div>', unsafe_allow_html=True)
    evaluation_task = f"請用台灣用語的繁體中文回答，並分別給模型1盒模型2模型輸出結果進行分數 0~10分。 請評估此兩個模型所輸出的結果:\n\n模型 1 ({model_options[model_1_id]}):\n{model_1_text}\n\n模型 2 ({model_options[model_2_id]}):\n{model_2_text}\n\n並解釋評分的理由，以及哪一個模型的表現比較好。"
    evaluation_prompt_list = [evaluation_task]
    evaluation_output_placeholder = st.empty()  # Create placeholder for streaming output
    evaluation_text = ""
    for chunk in model_3.stream(input=evaluation_prompt_list):
        evaluation_text += chunk  # Accumulate chunk text
        evaluation_output_placeholder.markdown(f'<div class="output-box">{evaluation_text}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
