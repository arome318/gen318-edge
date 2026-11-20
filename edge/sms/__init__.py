"""Offline SMS service for Gen318 Edge.

Turns local telemetry into plain-language SMS alerts for technicians and
community contacts, answers "POWER" status requests from anyone on the site,
and records technician fault labels (1 = real, 2 = false alarm, 3 = other).
Works with no internet: messages go through a GSM/LTE modem on the box.
"""
