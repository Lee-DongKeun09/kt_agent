from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from agent import collect_data, generate_report, save_report


def run_analysis(topic: str):
    if not topic.strip():
        yield "분석 대상을 입력해주세요.", ""
        return

    yield "🔍 데이터 수집 중...", ""

    data = collect_data(topic)

    yield "📝 보고서 생성 중...", ""

    report = generate_report(data)
    filepath = save_report(topic, report)

    yield f"✅ 완료! 저장됨: `{filepath}`", report


with gr.Blocks(title="시장 분석 에이전트", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 시장 동향 및 경쟁사 분석 에이전트")
    gr.Markdown("분석 대상을 입력하면 최신 시장 동향과 경쟁사 분석 보고서를 자동으로 생성합니다.")

    with gr.Row():
        with gr.Column(scale=3):
            topic_input = gr.Textbox(
                label="분석 대상",
                placeholder="예: 국내 OTT 시장, 테슬라 전기차, 네이버 AI",
                lines=1,
            )
        with gr.Column(scale=1):
            run_btn = gr.Button("분석 시작", variant="primary")

    status = gr.Textbox(label="진행 상태", interactive=False, lines=1)
    report_output = gr.Markdown(label="분석 보고서")

    run_btn.click(
        fn=run_analysis,
        inputs=topic_input,
        outputs=[status, report_output],
    )
    topic_input.submit(
        fn=run_analysis,
        inputs=topic_input,
        outputs=[status, report_output],
    )

if __name__ == "__main__":
    demo.launch()
