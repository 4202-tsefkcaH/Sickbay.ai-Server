from PIL import Image
import pytesseract
from pdf2image import convert_from_path
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
#images = convert_from_path('C:\\Users\\Admin\\Downloads\\Sample-filled-in-MR.pdf',poppler_path = r"C:\\Program Files\\poppler-24.02.0\\Library\\bin")
def pdf2text(path):
    images = convert_from_path(path,poppler_path = r"C:\\Program Files\\poppler-24.02.0\\Library\\bin")
    response = []
    for i in range(len(images)):
        temp_dict = {}
        temp_dict['page_no'] = i
        temp_dict['text'] = pytesseract.image_to_string(images[i])
        response.append(temp_dict)
    return response

print(pdf2text(r"D:\\Padhai\\1st Year\\2nd Sem\\Tutorial_sheet_1.pdf"))