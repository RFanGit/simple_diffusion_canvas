from image_process_utils import process_image

filename = 'static/saved_image.png'
model_number = 1
output_name = 'static/bad_result.png'
process_image(filename, model_number, output_name)

model_number = 4
output_name = 'static/good_result.png'
process_image(filename, model_number, output_name)