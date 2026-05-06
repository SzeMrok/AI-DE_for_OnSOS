### 1. How is an image converted into numbers?

Images are converted to numbers through feature extraction. Our feature extractor takes a 64x64 image and produces 12288 numerical features - basically flattening the pixel data into a single vector. The model then uses these numbers to learn patterns.

### 2. What does the model actually "see"?

The model "sees" numbers and learns which numbers go with which class labels. It's a pattern matching on the feature values.

### 3. Why can the model classify random or unknown images as one of the known classes?

Because the model is forced to pick one of the three classes (forest, river, residential). It always outputs the class it thinks is most likely, even if the input is completely random or from a class it's never seen before.

### 4. Why can this behavior be dangerous in real AI systems?

Because the model can make wrong predictions without any warning. In applications like medical diagnosis this could cause serious harm. The model might classify something dangerous as safe with high confidence.

### 5. What would need to be improved before using such a system operationally?

We'd need confidence thresholds so low-confidence predictions get flagged.

### 6. Which model achieved the highest accuracy?

Random Forest and SVM both got 75% accuracy. KNN had 69.44% and Logistic Regression had 63.89%.

### 7. Which model was fastest to train?

KNN was by far the fastest - basically instant (0.00s).

### 8. Did the fastest model also achieve the best accuracy?

No, KNN was the fastest but only got 69.44% accuracy.

### 9. Did all models classify the same image in the same way?

Not always - on test images they mostly agreed. But on the EuroSAT Highway image (which they weren't trained on), Random Forest predicted "residential". On the random noise image, models made different guesses too.

### 10. What does the accuracy vs training time plot show?

Logistic Regression was slow (35.41s) and had the worst accuracy. SVM and Random Forest were fast and both got 75% accuracy. KNN was fastest but had lower accuracy.
