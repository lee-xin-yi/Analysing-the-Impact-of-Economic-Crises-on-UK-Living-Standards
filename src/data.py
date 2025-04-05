import pandas as pd

def csv_time_series_to_df(csv_path, column=0, date_pattern=r"^\d{4}\s[A-Za-z]{3}$"):
    """
    Reads a CSV file and trims the DataFrame to the first run of rows that match a specified date pattern.

    Parameters:
    - csv_path (str): Path to the CSV file.
    - column (int or str, optional): The column to check for date patterns. Can be an index (int) or column name (str).
      Default is 0 (the first column).
    - date_pattern (str, optional): A regex pattern to match the date format.
      Default is '^\d{4}\s[A-Za-z]{3}$' (matches a year followed by a 3-letter month abbreviation).

    Returns:
    - pd.DataFrame: A DataFrame that is trimmed to include only rows where the specified column matches the date pattern,
      starting from the first matching row to the last.

    Raises:
    - ValueError: If the specified column does not exist or the date pattern is not matched.
    """

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Ensure column is either an index or column name
    if isinstance(column, int):
        if column < 0 or column >= len(df.columns):
            raise ValueError("Column index is out of range.")
        column_data = df.iloc[:, column]
    elif isinstance(column, str):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        column_data = df[column]
    else:
        raise ValueError("The 'column' argument must be either an index or column name.")

    # Match the dates using the provided pattern
    matches = column_data.str.match(date_pattern)

    # If there are no matches, handle the case gracefully
    if matches.sum() == 0:
        raise ValueError(f"No dates matching the pattern '{date_pattern}' were found in column '{column}'.")

    # Find index of first True and last True
    start_index = matches[matches].index[0]
    end_index = matches[matches].index[-1]

    # Slice the dataframe based on the valid date range
    df = df.loc[start_index:end_index]

    return df

def xlsx_reader(data_path, filename, unwanted_tabs=None):
    """
    Reads an Excel file and extracts data from sheets, ignoring common non-data sheets
    such as index, cover sheet, table of contents, and notes. This function attempts to
    remove unwanted content but may not be perfect, and extracted tables may require
    further preprocessing.

    Parameters:
    - data_path (str): The path to the directory where the Excel file is located.
    - filename (str): The name of the Excel file to read.
    - unwanted_tabs (list of str, optional): A list of sheet names to ignore (case-insensitive).
      Default is ["index", "cover_sheet", "cover sheet", "table_of_contents", "notes", "note", "contents"].

    Returns:
    - dict: A dictionary where keys are sheet names and values are DataFrames containing the cleaned data.

    Notes:
    - The function attempts to ignore non-data sheets commonly found in ONS Excel files, but this
      approach may not work perfectly for all files, and additional preprocessing may be needed.
    """

    # Set default unwanted_tabs if not provided
    if unwanted_tabs is None:
        unwanted_tabs = ["index", "cover_sheet", "cover sheet", "table_of_contents", "notes", "note", "contents"]

    # Read all sheets from the Excel file
    df = pd.read_excel(data_path + filename, sheet_name=None)

    # Remove unwanted tabs, ignoring case
    df = {k:v for k,v in df.items() if not k.lower() in [tab.lower() for tab in unwanted_tabs]}

    # Remove reference to notes in each cell: value [.*] -> value
    for sheet in df:
        df[sheet] = df[sheet].replace(to_replace=r'\[.*\]', value='', regex=True)

    # For each sheet, remove every row prior to the first row with more than 1 value
    # (effectively removing header/title/notes)
    for sheet in df:
        # remove rows prior
        df[sheet] = df[sheet][df[sheet].apply(lambda x: x.count(), axis=1) > 1]
        # set first row as columns
        df[sheet].columns = df[sheet].iloc[0]
        # remove first row
        df[sheet] = df[sheet][1:]

    return df
