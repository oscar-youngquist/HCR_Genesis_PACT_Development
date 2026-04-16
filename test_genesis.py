import genesis as gs
import traceback
import sys

try:
    gs.init(backend=gs.cuda)
    print("1. gs.init done")
    sys.stdout.flush()

    scene = gs.Scene(show_viewer=False)
    print("2. Scene created")
    sys.stdout.flush()

    scene.add_entity(gs.morphs.Plane())
    print("3. Plane added")
    sys.stdout.flush()

    scene.build()
    print("4. Scene built")
    sys.stdout.flush()

    print("Genesis works!")
except Exception as e:
    traceback.print_exc()
    sys.stdout.flush()
