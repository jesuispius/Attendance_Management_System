# ==================================================================================================================== #
# Creator: Phuoc Nguyen
# Date: March 22, 2021
# Filename: Tool.py
# Description: This file will provide the functions supporting stuff regarding the system, validity, etc.
# ==================================================================================================================== #

# Modules and libraries
import os


# -----------------------------------------------------VALITITY------------------------------------------------------- #
# Function to check if input is numerical data or not
def checkInteger(input_data):
    """
    Function to check if input is numerical data or not
    :param input_data:
    :return: a boolean result
    """
    try:
        int(input_data)
        return True
    except ValueError:
        pass
    return False


# -------------------------------------------------WORKING DIRECTORY-------------------------------------------------- #
# Function to check if directory exists
def isDir(dirpath):
    """
    Function to check if directory exists in the system
    :param dirpath:
    :return: a boolean result
    """
    result = os.path.isdir(dirpath)
    return result


# Function to check if file exists
def isFile(filename):
    """
    Function to check if file exists in the system
    :param filename:
    :return: a boolean result
    """
    result = os.path.isfile(filename)
    return result


# Function to show all files (contents) in a specific directory
def showAllFiles(dirpath):
    """
    Function to show all files in a specific directory
    :param dirpath:
    :return list of files:
    """
    list_of_files = os.listdir(dirpath)
    return list_of_files


# Function to delete all the contents of a specific directory
def deleteContentsDir(folderpath):
    """
    Function to delete all contents (files) in a specific directory
    :param folderpath:
    :return nothing:
    """
    # Check if the folder path exists or not
    if isDir(folderpath):
        # Traverse through each file in the target folder
        for each_path in showAllFiles(folderpath):
            # Connect the folder path and filename together
            # ...to have a complete absolute path.
            completed_path = os.path.join(folderpath, each_path)

            try:
                # Remove the file
                if isFile(completed_path):
                    os.remove(completed_path)
                # In case, remove the folder
                elif isDir(completed_path):
                    os.rmdir(completed_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (completed_path, e))
    else:
        print("The file does not exist!")
