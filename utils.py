import os
import json
import glob
import re
import dotenv
import ipywidgets as widgets
from IPython.display import display, HTML
from pprint import pprint
from dotenv import load_dotenv, find_dotenv
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
from unstructured.chunking.title import chunk_by_title
from llama_index.core import Document
from collections import Counter, defaultdict
from trulens_eval import Tru
import pydantic
from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI
)


class Utils:
    
    def __init__(self) -> None:
        pass

    
    def get_api_key(self, prefix=None) -> str:
        """
        Gets API keys from dotenv to avoid exposing private keys. Format of API name is prefix_API_KEY.

        Args: 
            prefix --> takes vendor name as prefix (eg. OPENAI)
        Returns: API key as string
        """
        _ = load_dotenv(find_dotenv())  
        
        # Or handle potential errors if needed
        if not prefix:
            raise ValueError("Please specify a provider for the api key")

        suffix = "_API_KEY"
        api_key_var = f"{prefix.upper()}{suffix}"
        api_key = os.getenv(api_key_var)

        if not api_key:
            raise KeyError(f"{prefix} API key not found in environment")

        return api_key
    
    
    def load_json_line_by_line(self, filename: str) -> list[dict]:
        """
        Loads records from a JSON file where each line is a separate JSON object.

        Args:
            filename --> the name of the file to read from
        Returns:
            list of dictionaries containing the loaded records
        """
        records = []
        try:
            with open(filename, "r") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if line:  # Ensure the line is not empty
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON on line {line_number}: {line}")
                            print(e)
        except IOError as e:
            print(f"Error: Failed to read data from {filename}.")
            print(e)
            return None
        else:
            print(f"Data loaded successfully from {filename}")
            return records
        
        
    def record_generator(self, records) -> object:
        """
        Loads records one at a time to avoid memory issues.

        Args: 
            records --> list of dictionaries to save as JSON
        Yields: generator object 
        """
        for record in records:
            yield record


    def save_json_line_by_line(self, filename: str, records: list[dict]) -> None:
        """
        Saves records as JSON objects, one per line, to the specified file.

        Args: 
            filename --> format [name.json]
            records --> list of dictionaries to save as JSON
        Returns: None
        """
        # Delete the file if it exists
        if os.path.exists(filename):
            os.remove(filename)
        
        try:
            with open(filename, "w") as f:
                for record in self.record_generator(records):
                    json_object = json.dumps(record)
                    f.write(json_object + '\n')
        except IOError as e:
            print(f"Error: Failed to write data to {filename}.")
            print(e)
        else:
            print(f"Data saved successfully to {filename}")
            
    
class FileSelector:
    
    def __init__(self) -> None:
        pass 
    
    def find_json_with_prefix(self, json_list: list, file_prefix: str) -> str:
        """
        Helper function for get_file_type. Searches json files in folder for filename match. If match, returns filename as PDF.
        Else, returns filename as json.

        Args: 
            json_list --> list of json files in folder, if any
            file_prefix --> prefix of file that will be queried 
        Returns: filename as string
        """
        for filename in json_list:
            if file_prefix in filename:
                return filename
        return file_prefix + '.pdf'
    
    def get_file_type(self, json_list: list, file_prefix: str) -> tuple[str, str]:
        """
        Helper function for get_file. Returns filename as either .pdf or .json; returns prefix as string or None.
        If filename already exists as json in folder, assigns NoneType to file_prefix as signal to skip preprocessing.

        Args: 
            json_list --> list of json files in folder, if any
            file_prefix --> prefix of file that will be queried
        Returns: tuple of filename and file_prefix
        """
        if not json_list:
            print(f'After processing, the parsed file will be saved as {file_prefix}.json in the current folder.')
            filename = f"{file_prefix}.pdf"
            return filename, file_prefix
        else:
            filename = self.find_json_with_prefix(json_list, file_prefix)
            if filename.endswith('.json'):
                file_prefix = None
            return filename, file_prefix

    def get_file(self) -> tuple[str, str]:
        """
        Creates a UI for selecting PDF user wants to parse.

        Returns: tuple of filename and file_prefix for later use in preprocessing logic.
        """
        pdf_list = glob.glob('*.pdf')
        json_list = glob.glob('*.json')

        if not pdf_list:
            print('Please upload a PDF to parse.')
            return None, None

        pdf_dropdown = widgets.Dropdown(
            options=pdf_list,
            description='Select PDF:',
        )

        proceed_dropdown = widgets.Dropdown(
            options=[('Yes', 1), ('No', 0)],
            description='Parse?',
        )

        parse_button = widgets.Button(description='Execute')
        output = widgets.Output()

        def on_button_clicked(b):
            with output:
                output.clear_output()
                filename = pdf_dropdown.value
                proceed = proceed_dropdown.value
                if proceed == 1:
                    file_prefix = filename.split('.')[0]
                    print(f'You selected {filename} to parse.')
                    result = self.get_file_type(json_list, file_prefix)
                    if result:
                        filename, file_prefix = result
                        if not file_prefix:
                            print(f'PDF already parsed. {filename} will be indexed.')
                        else: 
                            print(f'Selected file {filename} will be parsed and saved as json with prefix: {file_prefix}.')
                    else:
                        print('Error processing the file type.')
                elif proceed == 0:
                    print('Parsing aborted.')
                else:
                    print('Invalid input. Please select Yes or No.')
                return result

        parse_button.on_click(on_button_clicked)
        display(pdf_dropdown, proceed_dropdown, parse_button, output)
        result = on_button_clicked(self)
        return result
    
    
class Preprocess:
    
    def __init__(self, filename: str) -> None:
        self.filename = filename
    
    def read_file(self) -> object:
        """
        Creates an instance of the Files class from unstructured.io used to wrap the file's content and metadata, 
        which is then passed as part of the partition parameters in the request to the Unstructured API

        Returns: File object
        """
        with open(self.filename, "rb") as f:
            files = shared.Files(
                content=f.read(),
                file_name=self.filename
            )   
        return files
    
    
    def partition_file(self, files: object, strategy='auto', model_name="") -> object:
        """
        The PartitionParameters class is used to define the parameters for a partition request.

        Args: 
            files --> file content and metadata wrapped in a File object
            strategy --> strategy for partitioning document ('auto','fast', 'hi_res')
            model_name --> name of document layout detection model (only relevant to hi_res strategy)
        Returns: request, which is instructions to the API on how to parse 
        """
        infer_table=True
        if not model_name:
            infer_table=False
            
        req = shared.PartitionParameters(
            files=files,
            strategy=strategy,
            hi_res_model_name=model_name,
            pdf_infer_table_structure=infer_table,
            skip_infer_table_types=[],
        )
        return req
        
    
    def get_structured_text(self, client: object, req: object) -> tuple[list[dict], object]:
        """
        Produces parsed file in two forms for easy inspection.

        Args: 
            client --> UnstructuredClient is a client provided by unstructured for interacting with their API
            req --> instructions to the API on how to parse the file
        Returns: parsed file in two forms: tuple of records (list of dictionaries) and Element objects 
        """
        try:
            # use unstructured's API to process model-based work load (generating bounding boxes and extracting text)
            resp = client.general.partition(req)
            records = resp.elements
            elements = dict_to_elements(records)
            return records, elements
        except SDKError as e:
            print(e)
            
    
    
    def add_parent_to_metadata(self, elements: list[object], section_ids: dict) -> list[dict]:
        """
        Converts elements to records and adds section to metadata.

        Args: 
            elements --> filtered list off Element objects
            section_ids --> dictionary output from get_section_id_dict method of Inspect class
        Returns: list of dictionaries 
        """
        records = []
        for element in elements:
            record = element.to_dict()
            parent_id = record['metadata'].get('parent_id')
            section = section_ids.get(parent_id, "")
            record['metadata']['section'] = section
            records.append(record)
        return records
    
    
    def json_to_doc(self, data: list[dict]) -> object:
        """
        Converts records to elements, chunks by title, then concolidates into one Document object.

        Args: 
            data --> list of dictionaries (records)
        Returns: document object 
        """
        # convert json to Element objects
        elements = dict_to_elements(data)

        # chunk elements by title
        chunks = chunk_by_title(
            elements,
            combine_text_under_n_chars=100,
            max_characters=3000,
        )

        document = Document(text='\n\n'.join([chunk.text for chunk in chunks]))
        
        return document
            
         

class Inspect:
    
    def __init__(self, records: list[dict], elements: list[object]) -> None:
        pass
        self.records = records
        self.elements = elements
        
    
    def count_elements(self, elements=None) -> list[tuple]:
        """
        Counts number of elements that belong to each category. Useful for comparing count before and after filtering.
        If no elements list is given, it will take the original elements list generated from parsing.
        If filtering operations have been done, pass the modified elements list.

        Args: 
            elements --> list[object] (optional)
        Returns: List of tuples (category, count)
        """
        lmts = self.elements if elements is None else elements
        categories = [element.category for element in lmts]
        return Counter(categories).most_common()
    
    
    def inspect_record_type(self, type_string, max_items=None) -> list[dict]:
        """
        Allows for inspection of any specific record type available from count_elements (eg. Title, Table, etc).
        Also, helper function for get_section_id_dict.

        Args: 
            type_string --> string of element category
            max_items --> integer declaring max number of records to return (optional)
        Returns: list of dictionaries 
        """
        # filter records with matching type
        matching_records = [record for record in self.records if record['type'].lower() == type_string.lower()]

        # Option 1: Return all matching records (if desired)

        # Option 2: Return a slice with max_items (if needed)
        if max_items is not None:
            return matching_records[:max_items]
        else:
            return matching_records
        
    
    
    def get_section_id_dict(self, type_string='Title') -> dict:
        """
        Gets the unique element IDs for each parent. Also, helper function for get_references_and_header_id.

        Args: 
            type_string --> set to 'Title' as this in the name of the parent in unstructured.io
        Returns: dictionary {element_id: Title}
        """
        section_ids = {}
        
        section_records = self.inspect_record_type(type_string)
        section_titles = [record['text'] for record in section_records]
        for record in section_records:
            for title in section_titles:
                if record['text'] == title and record['type'] == type_string:
                    section_ids[record['element_id']] = title
                    break
                
        return section_ids
    
    
    def get_references_and_header_id(self, records) -> tuple[str, str]:
        """
        retrieves unique element ID for references and header sections for filtering.

        Args: 
            records --> list of dictionaries
        Returns: tuple (header_id, references_id) 
        """
        
        section_ids = self.get_section_id_dict()
        
        # get references id from get_section_id_dict method 
        references_id = [k for k, v in section_ids.items() if v.lower() == 'references'][0]
        
        # get header id from records 
        header_id = [record['element_id'] for record in records if record['type'].lower() == 'header'][0]
        
        return header_id, references_id
    
    
    def count_child_records(self, records=None) -> defaultdict:
        """
        Helper function for print_child_records. Creates a defaultdict to store parent-child relationships and counts.

        Args:
            records --> list of dictionaries
        Returns:
            A defaultdict where the key is a tuple (parent id, child record type) and the value is a dictionary:
                - 'num_children': The count of child elements that have this element as their parent (int, initially 0).
        """
        parent_child_counts = defaultdict(lambda: {'num_children': 0})

        recs = self.records if records is None else records
        
        for record in recs:
    
            parent_id = record['metadata'].get('parent_id', None)  # Handle potential missing parent_id
            record_type = record['type']

            if parent_id:
                # Existing parent, use (parent_id, parent_type) as key
                parent_child_counts[(parent_id, record_type)]['num_children'] += 1

        return parent_child_counts
    
    
    def print_child_records(self, records=None) -> None:
        """
        Generates a neat printout of parent and child relationship details.

        Args: 
            records --> list of dictionaries
        Returns: None 
        """
        show_text = True
        indent = '    '
        child_counts = self.count_child_records(records)
        
        for key, value in child_counts.items():
            parent_id = key[0]
            child_type = key[1]
            num_children = value['num_children']
            for record in self.records:
                if parent_id == record.get('element_id'):
                    parent_type = record.get('type')
                    parent_text = record.get('text')
                    if show_text:
                        print(f"PARENT TYPE: {parent_type}, PARENT ID: {parent_id}\nPARENT TEXT: {parent_text}\n{indent}CHILD TYPE: {child_type} --> NUMBER OF CHILDREN: {num_children}\n")
                    else:    
                        print(f"PARENT TYPE: {parent_type}, PARENT ID: {parent_id}\n{indent}CHILD TYPE: {child_type} --> NUMBER OF CHILDREN: {num_children}\n")
                        break
                    

                    
class QueryInterface:
    '''This setup allows users to enter a query in the text input widget, submit it using the button, 
    and see the response displayed in the output widget.'''
    def __init__(self, query_engine: object) -> None:
        self.query_engine = query_engine
        
        # Create widgets
        self.query_input = widgets.Text(
            value='',
            placeholder='Enter your query',
            description='Query:',
            disabled=False,
            layout=widgets.Layout(width='100%')  # Adjust the width to 100%
        )

        self.submit_button = widgets.Button(
            description='Submit',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click to submit query',
            icon='check'  
        )

        self.output = widgets.Output()

        # Attach the button click event to the handler
        self.submit_button.on_click(self.on_button_clicked)

    def on_button_clicked(self, b) -> None:
        with self.output:
            self.output.clear_output()  # Clear previous output
            query_string = self.query_input.value
            if query_string:
                response = self.query_engine.query(query_string)
                # Clean up response using regex
                clean_response = self.clean_response(response)
                display(HTML(f"<div style='width: 100%; word-wrap: break-word;'>{clean_response}</div>"))
            else:
                display(HTML("<div style='color: red;'>Please enter a query.</div>"))

    def clean_response(self, response: object) -> str:
        response = str(response)
        # response = re.sub(r'INFO:.*?OK"\n', '', response)
        match = re.search(r"\('([^']+)'(?: '[^']+')*\)", response)
        if match:
            # Join the text segments
            clean_text = ''.join(match.groups())
            return clean_text
        else:
            return response
        
    def display(self) -> None:
        display(self.query_input, self.submit_button, self.output)

####################### WIP ############################
class Eval:
    
    def __init__(self) -> None:
        pass
    
    def get_prebuilt_trulens_recorder(self, query_engine, app_id) -> object:
        tru_recorder = TruLlama(
            query_engine,
            app_id=app_id
        )
        
        return tru_recorder