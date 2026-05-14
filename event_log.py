"""Append-only JSON-serializable event log. No wall-clock timestamps."""

import json


class EventLog:
    def __init__(self):
        self._events = []
        self._step = 0

    def append(self, event):
        event["step"] = self._step
        self._step += 1
        self._events.append(event)

    def to_list(self):
        return list(self._events)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._events, f, indent=2)

    def total_cost(self):
        return sum(e.get("cost", 0.0) for e in self._events)

    def total_cost_by_type(self, event_type):
        return sum(
            e.get("cost", 0.0) for e in self._events if e.get("event") == event_type
        )

    def __len__(self):
        return len(self._events)

    def __iter__(self):
        return iter(self._events)

    def __getitem__(self, idx):
        return self._events[idx]
