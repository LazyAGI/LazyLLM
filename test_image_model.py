import lazyllm


llm = lazyllm.OnlineModule(source='doubao',
                            # model='Qwen/Qwen-Image-Edit-2509',
                            function='image_editing' )
                            # type='image_editing')
# lazyllm.WebModule(llm, port=23334, files_target=llm).start().wait()
# prompt = "融合这几张图片"
prompt = "在图片中添加随机数字,生成多张不同效果的图片"

ref_img1 = "D:\\template.png"
ref_img2 = "D:\\tmp0cdlfaeq.png"
ref_img3 = "D:\\tmp1fe3xyhj.jpg"
ref_img4 = "D:\\tmp1gcvqmmq.png"

response = llm(input = prompt,files = [ref_img1], n = 2)
print(response)












