from src.vision.create_gradcam import (
    load_class_names,
    load_model,
    load_image,
    predict,
    create_cam_by_method,
    visualize_multiple_cams
)


def main():
    class_names = load_class_names()
    model = load_model(class_names)
    image_path = ("data/processed/images/test/river/river_0000.jpg")
    image, tensor = load_image(image_path)
    predict(model, tensor, class_names)
    
    methods = ["GradCAM", "HiResCAM", "EigenCAM", "LayerCAM"]
    heatmaps = {}
    
    for method_name in methods:
        heatmaps[method_name] = create_cam_by_method(model, tensor, method_name)
    
    visualize_multiple_cams(image, heatmaps)
    
    
if __name__ == "__main__":
    main()