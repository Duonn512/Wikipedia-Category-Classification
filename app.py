import torch
import gradio as gr
from src.utils.preprocess import custom_transform

from src.models.bigru import BiGRU
from src.data.w2v_loader import W2VLoader
import requests
from lxml import html


LABELS = ["Khoa học tự nhiên","Khoa học xã hội","Kỹ thuật","Văn hóa"]
w2v_loader = W2VLoader('data/wiki.vi.model.bin.gz')
w2v_model = w2v_loader.get_model()
VOCAB_SIZE  = w2v_loader.vocab_size
EMBEDDING_DIM = w2v_loader.embedding_dim

TX = 80 # seq_length

model = BiGRU(input_size=EMBEDDING_DIM, 
                   hidden_size=256, 
                   num_classes=4,
                   embed_model=w2v_model,
                   num_layers=1,
                   dropout_prob=0.5)
model.load_state_dict(torch.load(f = 'saved_models/bigru.pth', map_location=torch.device('cpu')))

def crawl(url):
    response = requests.get(url)
    tree = html.fromstring(response.content)
    
    # Get content
    content_parts = []
    paragraphs = tree.xpath('//*[@id="mw-content-text"]/div[1]/p[1]//text()')
    if paragraphs and paragraphs not in [[' \n'],['\n'],['\n\n']] :
        for node in paragraphs :
            if node.strip():
                content_parts.append(node.strip())
    else:
        paragraphs = tree.xpath('//*[@id="mw-content-text"]/div[1]/p[2]//text()')
        for node in paragraphs:
            if node.strip():
                content_parts.append(node.strip())

    content = ' '.join(content_parts)
    
    # Get title
    title = tree.xpath('//h1[@id="firstHeading"]//text()')
    if title:
        title = title[0]
    else:
        title = None

    return {'title': title, 'content': content}

def get_topic(input_data):
    if input_data.startswith('http://') or input_data.startswith('https://'):
        text = crawl(input_data)['content']
    else:
        text = input_data
    
    processed_text = custom_transform(text, w2v_model, TX)
    
    device = next(model.parameters()).device
    processed_text = processed_text.to(device).unsqueeze(0)
    
    with torch.no_grad():
        probs = model(processed_text).squeeze(0)
    
    values, indices = torch.topk(probs, k=4)
        
    return {LABELS[i]: v.item() for i, v in zip(indices, values)}

def main():
    interface = gr.Interface(
        fn=get_topic,
        inputs=gr.Textbox(lines=2, placeholder="Nhập URL hoặc văn bản tại đây"),
        outputs="label",
        title="Phân loại Chủ đề bài viết trên Wikipedia Tiếng Việt",

        theme=gr.themes.Soft(),
        examples=[
            ["https://vi.wikipedia.org/wiki/H%E1%BB%8Dc_s%C3%A2u"],
            ["https://vi.wikipedia.org/wiki/S%C3%A9c"]
        ],
        allow_flagging="never",
        description="""
            Các chủ đề:
            1. Khoa học tự nhiên: Bao gồm các lĩnh vực như Địa chất học, Hóa học, Khoa học máy tính, Logic, Sinh học, Thiên văn học, Toán học, Vật lý học, Y học,...
            2. Khoa học xã hội: Bao gồm các lĩnh vực như Chính trị học, Công nghệ thông tin, Địa lý học, Địa lý học, Kinh tế học, Luật học, Ngôn ngữ học, Tâm lý học, Xã hội học,...
            3. Kỹ thuật: Bao gồm các lĩnh vực như Công nghệ thông tin, Cơ khí, Điện tử, Điện tử viễn thông, Hóa học, Kỹ thuật hóa học, Kỹ thuật môi trường, Kỹ thuật xây dựng, Kỹ thuật y sinh, Máy tính, Ô tô, Xây dựng,...
            4. Văn hóa: Bao gồm các lĩnh vực như Âm nhạc, Báo chí, Điện ảnh, Hội họa, Khoa học văn học, Lịch sử, Mỹ thuật, Ngôn ngữ học, Sân khấu, Thơ, Thể thao, Văn học, Văn hóa, Xã hội học,...
        """
    )
    
    interface.launch(share=True)

if __name__ == '__main__':
    main()