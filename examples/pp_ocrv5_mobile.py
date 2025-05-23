import lazyllm

if __name__ == "__main__":
    pp_ocr = lazyllm.TrainableModule(base_model="PP-OCRv5_mobile")
    pp_ocr.update_server()
    res = pp_ocr("path/to/pdf")
    for line in res:
        print(line.get_text())
