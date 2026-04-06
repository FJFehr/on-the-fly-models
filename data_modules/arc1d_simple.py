"""Shared utilities for ARC1D data modules."""


def filter_split(split, task_categories: list[str] | None, task_ids: list[int] | None):
    allowed_categories = set(task_categories) if task_categories is not None else None
    allowed_task_ids = set(task_ids) if task_ids is not None else None

    if allowed_categories is None and allowed_task_ids is None:
        return split

    filtered_tasks = []
    for task in split:
        if allowed_categories is not None and task["task_category"] not in allowed_categories:
            continue
        if allowed_task_ids is not None and task["task_id"] not in allowed_task_ids:
            continue
        filtered_tasks.append(task)
    return filtered_tasks
