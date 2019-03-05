Hier ist der Ordner für unseren Autoencoder.

Erklärung der einzelnen Dateien:

Die Autoencoder Dateien sind ein paar der getesten AE Architekturen.

Die Autoencoder_Final_weights.h5 sind die finalen trainierten Parameter. (Encoder nur für den encoder Teil)

Helper Functions sind einfach hilfs-Funktionen die oft verwendet worden.

game_recorder.py hat das Spiel aufgenommen und gespeichert. Hiermit sammelten wir einige Stunden Spielgeschehen.

process_recorded.py hat sozusagen das gesamte Trainieren des AE koordiniert, es greift auf die meisten Dateien zu.

Die grid search Variante erlaubt es viele verschiedene Architekturen und Parameter in nur einem aufwendigen durch lauf zu testen.
