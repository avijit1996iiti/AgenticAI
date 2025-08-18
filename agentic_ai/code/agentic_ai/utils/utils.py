import json


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_agentic_app_from_file(app, input_path: str, output_path: str) -> None:
    """
    Reads a message from input_path, runs it through the agentic app,
    and writes the result to output_path.

    Args:
        input_path (str): Path to the .txt file containing the input message.
        output_path (str): Path to save the output message.
    """
    # Read input message
    with open(input_path, "r") as f:
        input_message = f.read().strip()

    state = {"messages": [input_message]}

    # Run the app
    result = app.invoke(state)

    # Write output message
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Output written to: {output_path}")
