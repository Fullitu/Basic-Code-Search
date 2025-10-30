from math import log2

def RecallAt10(predictions: list[list], targets: list) -> float:
    """
    Calculate Recall@10 metric.
    
    Args:
        predictions (list of list): A list where each element is a list of predicted document IDs for a query.
        targets (list): A list of true document IDs for each query.
        
    Returns:
        float: The Recall@10 score.
    """
    correct_retrievals = 0
    total_queries = len(targets)
    
    for pred, target in zip(predictions, targets):
        if target in pred[:10]:
            correct_retrievals += 1
            
    recall_at_10 = correct_retrievals / total_queries if total_queries > 0 else 0.0
    return recall_at_10

def MRRAt10(predictions: list[list], targets: list) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) at 10 metric.
    
    Args:
        predictions (list of list): A list where each element is a list of predicted document IDs for a query.
        targets (list): A list of true document IDs for each query.
        
    Returns:
        float: The MRR@10 score.
    """
    total_reciprocal_rank = 0.0
    total_queries = len(targets)
    
    for pred, target in zip(predictions, targets):
        try:
            rank = pred.index(target) + 1
            if rank <= 10:
                total_reciprocal_rank += 1 / rank
        except ValueError:
            continue
            
    mrr_at_10 = total_reciprocal_rank / total_queries if total_queries > 0 else 0.0
    return mrr_at_10

def NDCGAt10(predictions: list[list], targets: list) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at 10 metric.
    
    Args:
        predictions (list of list): A list where each element is a list of predicted document IDs for a query.
        targets (list): A list of true document IDs for each query.
        
    Returns:
        float: The NDCG@10 score.
    """
    def dcg(relevances):
        return sum((2**rel - 1) / log2(idx + 2) for idx, rel in enumerate(relevances))
    
    total_ndcg = 0.0
    total_queries = len(targets)
    
    for pred, target in zip(predictions, targets):
        relevances = [1 if doc_id == target else 0 for doc_id in pred[:10]]
        ideal_relevances = [1] + [0]*9  # Ideal case: target is at rank 1
        
        actual_dcg = dcg(relevances)
        ideal_dcg = dcg(ideal_relevances)
        
        ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        total_ndcg += ndcg
            
    ndcg_at_10 = total_ndcg / total_queries if total_queries > 0 else 0.0
    return ndcg_at_10