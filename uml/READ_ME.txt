NATS UML diagrams — draw.io edition

OPEN FIRST: NATS_UML_Drawio.html
This collection shows all three diagrams and contains buttons to open them
in draw.io. Its diagrams display offline; the editor needs internet access.

EDITABLE DOCUMENT: NATS_Simulator_UML.drawio
Three page tabs: Class diagram, Aircraft states, One journey.
Open https://app.diagrams.net/ and use File > Open from > Device.
Alternatively use the gallery's Open all three in draw.io button.
Save your edits with File > Save as, choosing a local device file.

Each shape, text label and connector is editable. Class compartments are
children of the outer class, so they move together. Major connections are
attached to the classes/states. Some diagram labels are independent text
objects; adjust these if you reposition their connectors.

Individual .drawio files, PNG previews and SVG vector images are included.
The previews are generated from the same layout definition as the draw.io
files. SVG images remain crisp when enlarged. Diagram content does not
require colour to interpret; the highlights are only visual aids.

CHANGES FROM THE PLANTUML VERSION
Aligned boxes, larger text, horizontal aircraft lifecycle, short connector
labels, three visual sections in the sequence diagram. Eight classes and
all 32 sequence interactions remain. The route's origin and destination
are separate named relationships. Long implementation notes are kept out
of the drawings. There is no change to the SimPy code.

MODEL NOTES
This is the proposed revised simulator, not a claim that the starter code
already implements it. Runtime quantities are dimensionless. Fleets and
networks remain configurable. An aircraft occupies its stand throughout
unloading, charger waiting, charging, boarding and departure waiting.
Only Charging consumes a charger. Reservations must account for future
stand releases; initially full ports can require coordinated departures.
The diagram is not proof that the scheduler avoids all deadlocks.
Aircraft activity states alone are not the M/M/1 Markov chain.

OFFICIAL EDITOR AND DOCUMENTATION
https://app.diagrams.net/
https://www.drawio.com/docs/diagram-types/uml/
