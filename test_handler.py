from handler import Handler

def test_inference():
    handler = Handler()

    input_data = "what is the smartest dog?"
    processed_data = handler.preprocess(input_data)
    output_data = handler.inference(processed_data)
    decoded_data = handler.postprocess(output_data)
    print(decoded_data)

test_inference()