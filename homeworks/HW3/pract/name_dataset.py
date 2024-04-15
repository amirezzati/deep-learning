import subprocess
import unicodedata
import string
import torch
import random
from io import open
import glob
import os

class Dataset:
    def __init__(self):
        if not (os.path.exists("data.zip") and os.path.exists("data")):
            # Run wget command
            wget_process = subprocess.Popen(['wget', 'https://download.pytorch.org/tutorial/data.zip'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            wget_output, wget_error = wget_process.communicate()

            # Check for wget errors
            if wget_error:
                print("Error occurred while downloading data.zip:", wget_error.decode())
            else:
                print("Downloaded data.zip successfully.")

            # Run unzip command
            unzip_process = subprocess.Popen(['unzip', 'data.zip'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            unzip_output, unzip_error = unzip_process.communicate()

            # Check for unzip errors
            if unzip_error:
                print("Error occurred while unzipping data.zip:", unzip_error.decode())
            else:
                print("Unzipped data.zip successfully.")
        else:
            print("Files already exist, skipping download and extraction.")

        def findFiles(path): 
            return glob.glob(path)

        all_letters = string.ascii_letters + " .,;'"
        n_letters = len(all_letters)

        # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
        def unicodeToAscii(s):
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
                and c in all_letters
            )


        # Build the category_lines dictionary, a list of names per language
        category_lines = {}
        all_categories = []

        # Read a file and split into lines
        def readLines(filename):
            lines = open(filename, encoding='utf-8').read().strip().split('\n')
            return [unicodeToAscii(line) for line in lines]

        for filename in findFiles('data/names/*.txt'):
            category = os.path.splitext(os.path.basename(filename))[0]
            all_categories.append(category)
            lines = readLines(filename)
            category_lines[category] = lines

        self.n_categories = len(all_categories)
        self.all_categories = all_categories
        self.category_lines = category_lines
        self.n_letters = n_letters
        self.all_letters = all_letters

    def letterToIndex(self, letter):
        return self.all_letters.find(letter)

    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    def letterToTensor(self, letter):
        tensor = torch.zeros(1, self.n_letters)
        tensor[0][self.letterToIndex(letter)] = 1
        return tensor

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def lineToTensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor

    def categoryFromOutput(self, output):
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return self.all_categories[category_i], category_i

    def randomChoice(self, l):
        return l[random.randint(0, len(l) - 1)]

    def randomTrainingExample(self, ):
        category = self.randomChoice(self.all_categories)
        line = self.randomChoice(self.category_lines[category])
        category_tensor = torch.tensor([self.all_categories.index(category)], dtype=torch.long)
        line_tensor = self.lineToTensor(line)
        return category, line, category_tensor, line_tensor