# load 3rd party repos
import sys
from pathlib import Path
from configparser import ConfigParser

project_path = Path(__file__).parent.parent.parent.parent

# Add LibBrain to path to import DataLoaders
libbrain_path = project_path / "LibBrain"
if str(libbrain_path) not in sys.path:
    sys.path.insert(0, str(libbrain_path))

# Add WholeBrain to path to import DataLoaders
wholebrain_path = project_path / "WholeBrain"
if str(wholebrain_path) not in sys.path:
    sys.path.insert(0, str(wholebrain_path))
