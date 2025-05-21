import os
import base64

import pandas as pd
def generate_html_with_images(differences, output_file='error_report.html'):
    html = """
    <html>
    <head><title>Error Report</title></head>
    <body>
    <h1>错误分类结果</h1>
    <table border="1" style="border-collapse: collapse;">
    
    <tr><th>Index</th><th>Image1</th><th>Image1</th><th>Image2</th><th>Image2</th><th>Predict</th><th>Score</th><th>True Label</th></tr>
    """

    for index, row in differences.iterrows():
        idx = row['index']
        img1 = row['img1']
        img2 = row['img2']
        predict = row['predict']
        score = round(float(row['score']), 4)
        true_label = row['trueLabel']


        file1 = os.path.join('./error_image' ,img1.replace('/','-'))
        file2 = os.path.join('./error_image' ,img2.replace('/','-'))

        # 将图片编码为 base64，嵌入到 HTML 中
        with open(file1, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
            image_data1 = f"data:image/jpeg;base64,{encoded_string}"

        with open(file2, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
            image_data2 = f"data:image/jpeg;base64,{encoded_string}"
        html += f"""<tr>
                                <td align="center">{idx}</td>
                                <td align="center">{img1}</td>
                                <td><img src="{image_data1}" width="100"></td>
                                <td align="center">{img2}</td>
                                <td><img src="{image_data2}" width="100"></td>
                                <td align="center">{predict}</td>
                                <td align="center">{score}</td>
                                <td align="center">{true_label}</td>
                            </tr>
                            """
    html += "</table></body></html>"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ 已生成 HTML 报告：{output_file}")

if __name__ == '__main__':
    differences = pd.read_csv('./difference.csv')
    generate_html_with_images(differences)