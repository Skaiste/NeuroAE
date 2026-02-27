# load 3rd party repos
import sys
from pathlib import Path
from configparser import ConfigParser

src_path = Path(__file__).parent.parent.parent
project_path = src_path.parent
trd_party_path = src_path / "3rd_party"

# Add LibBrain to path to import DataLoaders
libbrain_path = trd_party_path / "LibBrain"
if str(libbrain_path) not in sys.path:
    sys.path.insert(0, str(libbrain_path))
