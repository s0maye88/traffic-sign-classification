{
 "cells": [
  {
   "metadata": {},
   "cell_type": "markdown",
   "source": "#  Demo"
  },
  {
   "metadata": {
    "ExecuteTime": {
     "end_time": "2025-11-03T14:50:31.909541Z",
     "start_time": "2025-11-03T14:50:13.006535Z"
    }
   },
   "cell_type": "code",
   "source": [
    "from tensorflow.keras.models import load_model\n",
    "import numpy as np\n",
    "import cv2\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "\n",
    "# Load the trained model\n",
    "model = load_model('traffic_classifier.h5')\n",
    "\n",
    "# Label Overview\n",
    "labels = { 0:'Speed limit (20km/h)',\n",
    "            1:'Speed limit (30km/h)',\n",
    "            2:'Speed limit (50km/h)',\n",
    "            3:'Speed limit (60km/h)',\n",
    "            4:'Speed limit (70km/h)',\n",
    "            5:'Speed limit (80km/h)',\n",
    "            6:'End of speed limit (80km/h)',\n",
    "            7:'Speed limit (100km/h)',\n",
    "            8:'Speed limit (120km/h)',\n",
    "            9:'No passing',\n",
    "            10:'No passing veh over 3.5 tons',\n",
    "            11:'Right-of-way at intersection',\n",
    "            12:'Priority road',\n",
    "            13:'Yield',\n",
    "            14:'Stop',\n",
    "            15:'No vehicles',\n",
    "            16:'Veh > 3.5 tons prohibited',\n",
    "            17:'No entry',\n",
    "            18:'General caution',\n",
    "            19:'Dangerous curve left',\n",
    "            20:'Dangerous curve right',\n",
    "            21:'Double curve',\n",
    "            22:'Bumpy road',\n",
    "            23:'Slippery road',\n",
    "            24:'Road narrows on the right',\n",
    "            25:'Road work',\n",
    "            26:'Traffic signals',\n",
    "            27:'Pedestrians',\n",
    "            28:'Children crossing',\n",
    "            29:'Bicycles crossing',\n",
    "            30:'Beware of ice/snow',\n",
    "            31:'Wild animals crossing',\n",
    "            32:'End speed + passing limits',\n",
    "            33:'Turn right ahead',\n",
    "            34:'Turn left ahead',\n",
    "            35:'Ahead only',\n",
    "            36:'Go straight or right',\n",
    "            37:'Go straight or left',\n",
    "            38:'Keep right',\n",
    "            39:'Keep left',\n",
    "            40:'Roundabout mandatory',\n",
    "            41:'End of no passing',\n",
    "            42:'End no passing veh > 3.5 tons' }\n"
   ],
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    }
   ],
   "execution_count": 1
  },
  {
   "cell_type": "code",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2025-11-03T14:50:40.518567Z",
     "start_time": "2025-11-03T14:50:39.970986Z"
    }
   },
   "source": [
    "# Simple test function\n",
    "def test_image(image_path):\n",
    "    image = cv2.imread(image_path)\n",
    "    if image is None:\n",
    "        print(\"Image not found.\")\n",
    "        return\n",
    "    image = cv2.resize(image, (30, 30))\n",
    "    image = np.expand_dims(image / 255.0, axis=0)\n",
    "    pred = model.predict(image)\n",
    "    class_id = np.argmax(pred)\n",
    "    label = labels[class_id]\n",
    "    print(\"Predicted:\", label)\n",
    "\n",
    "    # Show image\n",
    "    img_show = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)\n",
    "    plt.imshow(img_show)\n",
    "    plt.title(f\"Predicted: {label}\")\n",
    "    plt.axis('off')\n",
    "    plt.show()\n",
    "\n",
    "# Test on one image\n",
    "test_image(r'C:\\Users\\somay\\Downloads\\Autonomos\\Test\\00150.png')"
   ],
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001B[1m1/1\u001B[0m \u001B[32m━━━━━━━━━━━━━━━━━━━━\u001B[0m\u001B[37m\u001B[0m \u001B[1m0s\u001B[0m 290ms/step\n",
      "Predicted: Speed limit (100km/h)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ],
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYwAAAGbCAYAAADX6qdpAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjUsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvWftoOwAAAAlwSFlzAAAPYQAAD2EBqD+naQAAPRVJREFUeJzt3Qm0JHdd9vGqrt7uNltmMpM9JEAIEEiMCqKCG/oqeNxQccejIooIbii4IOCCgqICKojixjkgLihuCEdUIL4SSCAQMslkm0wy+8y9c/dequs9/w4978ydep77vzWdEJnv55xA0v9b1bX/uvo+91dpURRFAgDAOmrr/QAAAAEFAwAQhYIBAIhCwQAARKFgAACiUDAAAFEoGACAKBQMAEAUCgYAIAoF47Po8ssvT573vOed/O//+I//SNI0Hf7/I3UZzyWx6x722S//8i+f/O8//dM/Hb527733jm1ZwvzDPM/WRz7ykaTZbCZ79+5NHglGx/xf//Vff1beP+zf6enpdX/u2LFjydTUVPLP//zPybnsnC0Yo5N69E+73U4e+9jHJj/2Yz+WHDp0KPnfJBzEp16wHknCRfP7v//7kyuvvHK4jXft2pU8/elPT17xild8thftf71f+7VfS9797ndvaJqf//mfT77jO74jueyyy04rIj/6oz+aXH/99Umj0Vi3MP3xH/9xcvXVVw/352Me85jkDW94Q+nPPfDAA8m3fdu3JVu2bEk2bdqUfMM3fENy9913J59t3/It35J83dd93YamOe+885If/MEfTH7xF38xOZedswVj5FWvelXyF3/xF8kb3/jG5GlPe1ryB3/wB8kXfdEXJcvLyw/7soQL6crKyvD/N1owXvnKVyaPNHfeeWdy3XXXJe9973uHF6mwjV/4whcOT77f+I3fSD5Xfc/3fM9wP556UT5bv/ALvzCc59kUjI9//OPJ+9///uQFL3jBGcfPW9/61mGhuOKKK+w83vzmNw8vnE94whOGhSKcKz/+4z9+xv5cXFxMvvzLvzz5z//8z+TlL3/58Pi8+eabk2c84xnDT+ufLb1eL3nf+96XPOtZz9rwtC94wQuSm266Kfn3f//35FxVT85xX/u1X5t8/ud//vDfw4kQLma//du/nfz93//98CJXZmlpaXh7Om61Wm34qe1zxetf//rhhSNcqNZePA8fPpx8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"
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "execution_count": 2
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
