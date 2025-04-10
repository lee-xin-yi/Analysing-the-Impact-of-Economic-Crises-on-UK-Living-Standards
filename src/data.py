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

def detect_date_column(df: pd.DataFrame) -> str:
    """
    Automatically detects the column in the DataFrame that contains date-like information,
    including quarterly data in the format "YYYY Qx".

    Parameters:
    - df (pd.DataFrame): The DataFrame to search for date columns.

    Returns:
    - str: The name of the detected date column, or None if no date column is found.
    """
    for column in df.columns:
        try:
            # Convert column to string if it's not already a string type
            if df[column].dtype != 'O':  # Not object type (string)
                df[column] = df[column].astype(str)

            # Check if column values are in "YYYY Qx" format (e.g., "2021 Q1")
            if df[column].str.contains(r'\d{4} Q[1-4]', na=False).any():
                return column  # Return the column name if it contains "YYYY Qx" format

            # Try to convert the column to datetime for standard date formats
            pd.to_datetime(df[column], errors='raise')
            return column  # Return the column name if it's a valid date column
        except (ValueError, TypeError):
            continue
    return None  # Return None if no valid date column is found

def convert_quarter_to_date(quarter_str: str) -> pd.Timestamp:
    """
    Converts a quarterly string (e.g., "2021 Q1") to the corresponding date (e.g., "2021-01-01").
    This function is flexible and can work with any quarterly string containing "YYYY Qx".

    Parameters:
    - quarter_str (str): The quarterly string (e.g., "2021 Q1").

    Returns:
    - pd.Timestamp: The corresponding datetime object (e.g., "2021-01-01").
    """
    # Split the string into year and quarter parts
    parts = quarter_str.split(" Q")
    if len(parts) == 2:
        year, quarter = parts
        # Handle the case where the string contains a valid year and quarter
        try:
            year = int(year)
            quarter = int(quarter)
            # Map quarters to the first month of each quarter
            quarter_months = {1: 1, 2: 4, 3: 7, 4: 10}
            if quarter in quarter_months:
                month = quarter_months[quarter]
                return pd.to_datetime(f"{year}-{month:02d}-01")
        except ValueError:
            pass  # If parsing fails, return NaT (could be invalid data)
    return pd.NaT  # Return NaT if the input string is not in a valid "YYYY Qx" format

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a DataFrame by stripping column names and converting date columns to datetime format.
    If a column is detected as a date, it is renamed to 'Date'.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The cleaned DataFrame.
    """
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    date_column = detect_date_column(df)

    if date_column:
        # Check if the column contains quarterly-like data (e.g., "YYYY Qx")
        if df[date_column].dtype == object and df[date_column].str.contains(r'\d{4} Q[1-4]', na=False).any():
            # Convert the "YYYY Qx" format to datetime dynamically
            df[date_column] = df[date_column].apply(convert_quarter_to_date)
        else:
            # For standard date formats, convert to datetime
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

        # Rename the date column to 'Date'
        df.rename(columns={date_column: 'Date'}, inplace=True)

    return df

def normalisation(df, normalization_type='z-score', columns=None):
    """
    Normalizes the numeric columns of a DataFrame using the specified normalization method.

    Args:
    - df (pd.DataFrame): The DataFrame to normalize.
    - normalization_type (str): The type of normalization to apply. Options are 'z-score' or 'min-max'. Default is 'z-score'.
    - columns (list or None): List of column names to normalize. If None, normalizes all numeric columns.

    Returns:
    - pd.DataFrame: DataFrame with normalized values.
    """
    # Select numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    if columns is not None:
        # Use provided columns, ensuring they are numeric
        numeric_columns = [col for col in columns if col in numeric_columns]

    if len(numeric_columns) == 0:
        raise ValueError("No numeric columns to normalize in the provided DataFrame.")

    # Select the normalization method
    if normalization_type == 'z-score':
        scaler = StandardScaler()
    elif normalization_type == 'min-max':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}. Supported types are 'z-score' and 'min-max'.")

    # Apply the scaler to the numeric columns
    normalized_data = scaler.fit_transform(df[numeric_columns])

    # Convert the normalized data back to a DataFrame
    normalized_df = df.copy()
    normalized_df[numeric_columns] = normalized_data

    return normalized_df

def preprocess_data(df: pd.DataFrame, freq: str = 'MS', method: str = 'ffill') -> pd.DataFrame:
    """
    Preprocesses a DataFrame by converting date formats and resampling it to a given frequency.

    This version supports various types of date formats and resampling methods.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - freq (str): The resampling frequency. Default is 'MS' (monthly start).
      Other options include 'D' (daily), 'M' (monthly), 'Y' (yearly), etc.
    - method (str): Method for filling missing data during resampling. Default is 'ffill' (forward fill).
      Other options: 'bfill' (backward fill), 'pad' (pad), 'nearest', etc.

    Returns:
        - pd.DataFrame: Preprocessed DataFrame with resampled data.
    """
    df = df.dropna(subset=['Date'])

    df_resampled = df.set_index('Date').resample(freq).apply(method).reset_index()

    return df_resampled

def merge_dataframes_on_date(*dfs):
    """
    Merges multiple dataframes on the 'Date' column.

    Parameters:
    - *dfs: Multiple DataFrames to merge.

    Returns:
    - pd.DataFrame: A single merged DataFrame on the 'Date' column.
    """
    # Preprocess each DataFrame before merging
    processed_dfs = [preprocess_data(df.copy()) for df in dfs]

    # Merge DataFrames on 'Date'
    merged_df = processed_dfs[0]
    for df in processed_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='Date', how='inner')

    return merged_df
