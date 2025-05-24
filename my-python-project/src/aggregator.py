# aggregator.py

def aggregate_data(data_sources):
    """
    Aggregates data from multiple sources.

    Parameters:
    data_sources (list): A list of data sources to aggregate.

    Returns:
    dict: A dictionary containing aggregated data.
    """
    aggregated_result = {}
    
    for source in data_sources:
        for key, value in source.items():
            if key in aggregated_result:
                aggregated_result[key] += value
            else:
                aggregated_result[key] = value

    return aggregated_result

def combine_results(results):
    """
    Combines results from different processes.

    Parameters:
    results (list): A list of results to combine.

    Returns:
    dict: A dictionary containing combined results.
    """
    combined_result = {}
    
    for result in results:
        for key, value in result.items():
            if key in combined_result:
                combined_result[key].append(value)
            else:
                combined_result[key] = [value]

    return combined_result