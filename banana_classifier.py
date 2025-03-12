import tkinter as tk  # GUI library for creating user interfaces
from tkinter import filedialog  # To open file dialog for selecting images
from tkinter.font import Font  # To set custom fonts for GUI elements
import torch  # PyTorch library for building and using deep learning models
import torchvision.transforms as transforms  # Image transformations for preprocessing
from PIL import Image, ImageTk  # To handle and display images
from torchvision import models  # Pretrained models from PyTorch
import torch.nn as nn  # To define and use neural network layers


# Define a custom CNN model based on ResNet50
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        # Load pretrained ResNet50 model with ImageNet weights
        self.model = models.resnet50(weights="IMAGENET1K_V1")
        # Freeze all layers to avoid updating their weights
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze the last two layers for fine-tuning
        for param in self.model.layer3.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        # Replace the final fully connected layer to match the number of classes
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),  # Add a hidden layer with 512 units
            nn.ReLU(),  # Activation function
            nn.Dropout(p=0.5),  # Dropout for regularization
            nn.Linear(512, num_classes)  # Final layer with output equal to number of classes
        )

    def forward(self, x):
        # Define the forward pass
        return self.model(x)


# Define the GUI application for banana disease classification
class ImageClassifierApp:
    def __init__(self, root):
        self.root = root  # Reference to the main Tkinter window
        self.root.title("Banana Disease Classifier")  # Set the window title
        self.root.geometry("1000x600")  # Set the window size

        # Create the main frame for the layout
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill="both", expand=True)

        # Create left and right frames for separating content
        self.left_frame = tk.Frame(self.main_frame, width=500, bg="white")
        self.right_frame = tk.Frame(self.main_frame, width=500, bg="white")
        self.left_frame.pack(side="left", fill="both", expand=True)
        self.right_frame.pack(side="right", fill="both", expand=True)

        # Label for displaying the uploaded image
        self.image_label = tk.Label(self.left_frame, bg="white")
        self.image_label.pack(pady=10)

        # Labels for displaying the disease name and confidence
        self.disease_label = tk.Label(self.left_frame, text="", font=("Arial", 16, "bold"), bg="white")
        self.disease_label.pack(pady=5)

        self.confidence_label = tk.Label(self.left_frame, text="", font=("Arial", 14), bg="white")
        self.confidence_label.pack(pady=5)

        # Text widget to display causes and prevention information
        self.result_text = tk.Text(self.right_frame, font=("Arial", 14), wrap="word", bg="white", height=25, width=60)
        self.result_text.pack(pady=20, padx=10)
        self.result_text.config(state="disabled")  # Make it read-only initially

        # Button for uploading an image
        self.upload_button = tk.Button(self.left_frame, text="Upload Image", command=self.upload_image, font=("Arial", 14))
        self.upload_button.pack(pady=10)

        # Load the pretrained model
        self.model = self.load_model()
        # Class names for prediction labels
        self.class_names = [
            "Banana Bract Mosaic Virus Disease",
            "Banana Healthy Leaf",
            "Banana Insect Pest Disease",
            "Banana Moko Disease",
            "Banana Panama Disease",
            "Banana Sigatoka Disease",
        ]

        # Dictionary to store causes and prevention info for diseases

        self.disease_info = {
            "Banana Bract Mosaic Virus Disease": {
                "Causes": [
                    "\n1] Banana Bract Mosaic Virus (BBrMV), which is transmitted by aphids, particularly by 'Pentalonia Nigronervosa'.\n",
                    "2] Aphids feed on the sap of infected plants and then transfer the virus to healthy plants.\n",
                    "3] Symptoms include mosaic-like streaks on the bracts (modified leaves) and distorted growth.\n",
                    "4] The virus can also be spread through infected planting materials.\n"

                ],
                "Preventions": [
                    "\n1) Use virus-free planting material.\n",
                    "2) Control aphid populations with insecticides or natural predators.\n",
                    "3) Remove and destroy infected plants to prevent the virus from spreading.\n"
                ],
            },
            "Banana Insect Pest Disease": {
                "Causes": [
                    "\nDamage caused by insects such as:\n",
                    "i] Banana weevils (Cosmopolites sordidus): Adults bore into the plant’s stem, weakening the plant and reducing yield.\n",
                    "ii] Nematodes: Microscopic worms that attack the roots, causing stunted growth and poor nutrient uptake.\n",
                    "iii] Aphids: These sap-sucking pests weaken plants and may also transmit viruses like Banana Bract Mosaic Virus.\n"

                ],
                "Preventions": [
                    "\n1) Apply insecticides to control pests.\n",
                    "2) Regularly inspect plants for early signs of infestation.\n",
                ],
            },
            "Banana Moko Disease": {
                "Causes": [
                    "\n1] Moko disease is caused by the bacterium 'Ralstonia solanacearum.'\n"
                    "\n2] It spreads through:\n",
                    "\ti] Infected tools used for pruning or harvesting.\n",
                    "\tii] Contaminated soil or water.\n",
                    "\tiii] Insects, that come into contact with infected flowers.\n",
                    "3] Symptoms include wilting of leaves, yellowing, and internal discoloration of the fruit and stem.\n"

                ],
                "Preventions": [
                    "\n1) Sanitize tools before use. \n",
                    "2) Avoid planting in areas with known infections. \n",
                    "3) Remove and destroy infected plants immediately. \n"
                ],
            },
            "Banana Panama Disease": {
                "Causes": [
                    "\n1] Caused by the soil-borne fungus 'Fusarium oxysporum cubense (Foc)'.\n",
                    "2] This disease attacks the banana plant’s roots and vascular system, cutting off water and nutrient flow.\n",
                    "3] There are different strains of the fungus, with Tropical Race 4 (TR4) being the most devastating.\n",
                    "4] It spreads through:",
                    "\ti] Contaminated soil or water.",
                    "\tii] Movement of infected plant material.",
                    "\tiii] Contaminated equipment.",
                    "\n5] Symptoms include yellowing of leaves, wilting, and black streaks inside the stem."

                ],
                "Preventions": [
                    "\n1) Use disease-resistant banana varieties. \n",
                    "2) Practice crop rotation and maintain good field sanitation. \n",
                ],
            },
            "Banana Sigatoka Disease": {
                "Causes": [
                    "\n1] Sigatoka is caused by fungal pathogens 'Mycosphaerella fijiensis' & 'Mycosphaerella musicola.' \n",
                    "2] The fungi infect banana leaves, forming dark streaks or spots that merge and lead to leaf death. \n",
                    "3] These fungi thrive in warm, humid environments and spread via windborne spores. \n"

                ],
                "Preventions": [
                    "\n1) Remove and destroy infected leaves.\n",
                    "2) Apply fungicides regularly as a preventive measure. \n",
                    "3) Ensure proper plant spacing to allow air circulation and reduce humidity. \n"
                ],
            },
        }

        # Function to handle image upload

    def upload_image(self):
        # Open a file dialog to select an image file
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.display_image(file_path)  # Display the uploaded image
            prediction, confidence = self.classify_image(file_path)  # Classify the image

            if confidence < 90:  # Check if confidence is low
                self.disease_label.config(text="Confidence low")
                self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
                self.display_result("", "")
            else:
                if prediction == "Banana Healthy Leaf":  # If the leaf is healthy
                    self.disease_label.config(text="Prediction: Banana Healthy Leaf\nThis is a healthy leaf")
                    self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
                    self.display_result("", "")  # No causes or prevention for a healthy leaf
                else:  # If a disease is detected
                    self.disease_label.config(text=f"Prediction: {prediction}")
                    self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
                    # Get disease information and display it
                    disease_info = self.disease_info.get(prediction, {})
                    causes = "\n".join(disease_info.get("Causes", []))
                    preventions = "\n".join(disease_info.get("Preventions", []))
                    self.display_result(causes, preventions)

        # Function to display the uploaded image in the GUI

    def display_image(self, file_path):
        image = Image.open(file_path)  # Open the image file
        image = image.resize((400, 400))  # Resize the image to fit in the left frame
        photo = ImageTk.PhotoImage(image)  # Convert the image for Tkinter
        self.image_label.config(image=photo)  # Set the image in the label
        self.image_label.image = photo  # Keep a reference to avoid garbage collection

        # Function to display causes and preventions in the text widget

    def display_result(self, causes, preventions):
        self.result_text.config(state="normal")  # Make the text widget editable
        self.result_text.delete("1.0", tk.END)  # Clear existing text
        if causes or preventions:
            self.result_text.insert("insert", "Causes:\n", "bold")  # Insert causes
            self.result_text.insert("insert", causes + "\n\n", "normal")
            self.result_text.insert("insert", "Preventions:\n", "bold")  # Insert preventions
            self.result_text.insert("insert", preventions, "normal")
        # Configure text styles
        self.result_text.tag_configure("bold", font=("Arial", 14, "bold"))
        self.result_text.tag_configure("normal", font=("Arial", 14))
        self.result_text.config(state="disabled")  # Make it read-only again

        # Function to load the pretrained model

    def load_model(self):
        model = CNNModel(num_classes=6)  # Create an instance of the CNNModel
        model.load_state_dict(
            torch.load(
                r"C:\Users\syhaw\Downloads\Mega-Project\Models\BDD_CNN_CLAHE_HBOA_Ci6.pth",
                map_location="cpu",  # Load the model on CPU
                weights_only=True
            )
        )
        model.eval()  # Set the model to evaluation mode
        model.to(torch.device("cpu"))  # Move the model to CPU
        return model

        # Function to preprocess an image before classification

    def preprocess_image(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
        ])
        image = Image.open(image_path).convert("RGB")  # Open the image and convert to RGB
        return transform(image).unsqueeze(0)  # Add a batch dimension

        # Function to classify the uploaded image

    def classify_image(self, image_path):
        image_tensor = self.preprocess_image(image_path)  # Preprocess the image
        outputs = self.model(image_tensor)  # Get the model's predictions
        probabilities = torch.softmax(outputs, dim=1)  # Convert to probabilities
        confidence, predicted_class = torch.max(probabilities, dim=1)  # Get the top prediction
        return self.class_names[predicted_class.item()], confidence.item() * 100  # Return the label and confidence


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
