import os
import cv2
import pytesseract

# Đường dẫn đến thư mục chứa ảnh
image_folder = 'D:\PBL5\ServerPBL5\PBL5_MainServers\PBL5_MainServer\output'

# Đường dẫn đến thư mục để lưu văn bản nhãn
output_folder = 'D:\PBL5\ServerPBL5\PBL5_MainServers\PBL5_MainServer\output_labels'

# Thiết lập ngôn ngữ cho Tesseract OCR (tùy chọn)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Duyệt qua các tệp tin trong thư mục ảnh
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Đường dẫn đầy đủ đến tệp tin ảnh
        image_path = os.path.join(image_folder, filename)
        
        # Đọc ảnh từ tệp tin
        image = cv2.imread(image_path)
        detected_text = pytesseract.image_to_string(image)
        
        # Sau khi có văn bản nhãn, lưu nó dưới dạng văn bản vào thư mục đầu ra
        output_text_file = os.path.splitext(filename)[0] + '.txt'  # Tên tệp tin văn bản
        output_path = os.path.join(output_folder, output_text_file)
        
        with open(output_path, 'w') as file:
            file.write(detected_text)
