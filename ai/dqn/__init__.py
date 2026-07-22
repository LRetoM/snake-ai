"""Deep Q-Learning (Phase 3b): EIN Netz, EIN Erfahrungs-Tagebuch, mehrere
gleichzeitig spielende Schlangen.

Diese KI ist voellig unabhaengig von der Neuroevolution (ai/evolution/). Geteilt
werden nur WERKZEUGE -- die Spiel-Engine (game/), die Wahrnehmung
(ai/perception.py) und das Netz-Grundgeruest (ai/torch_bridge.py) -- niemals
gelerntes Wissen. Die beiden Bots lernen komplett getrennt und koennen am Ende
fair verglichen werden.
"""
