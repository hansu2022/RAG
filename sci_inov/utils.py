import pandas as pd

def read_excel_data(file_path, sheet_name=0):
    """
    Reads data from an Excel file.

    Args:
        file_path (str): The path to the Excel file.
        sheet_name (str or int, optional): The name or index of the sheet to read.
                                         Defaults to 0 (the first sheet).

    Returns:
        pandas.DataFrame: A DataFrame containing the data from the Excel sheet.
                          Returns None if an error occurs.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the Excel file: {e}")
        return None

if __name__ == '__main__':
    # Example usage (assuming you have an 'experts.xlsx' file in the same directory)
    # Create a dummy Excel for testing if it doesn't exist
    try:
        pd.read_excel('experts.xlsx')
    except FileNotFoundError:
        print("Creating a dummy 'experts.xlsx' for testing.")
        dummy_data = {
            'name': ['Alice Wonderland', 'Bob The Builder'],
            'dept': ['Computer Science', 'Civil Engineering'],
            'lab': ['AI Lab', 'Construction Lab'],
            'title': ['Professor', 'Senior Engineer'],
            'research': ['Artificial Intelligence', 'Sustainable Construction'],
            'personal_page_addr': ['http://alice.edu', 'http://bob.com']
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_excel('experts.xlsx', index=False)
        print("Dummy 'experts.xlsx' created. Please populate it with actual data or ensure your target Excel exists.")


    excel_file = 'experts.xlsx' # Replace with your Excel file path
    data = read_excel_data(excel_file)

    if data is not None:
        print("Data read from Excel:")
        print(data) 