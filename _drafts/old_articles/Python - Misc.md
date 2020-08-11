#### Loop over all files in folder

```
import os
my_path = "your_path_here"
filenames = [entry for entry in os.scandir(my_path) if entry.is_file()]

for filename in filenames:
  with open(filename.path, "r") as file:
    # do something
```

More info about scandir in its [Github repo](https://github.com/benhoyt/scandir).
See also [os.walk](http://pythoncentral.io/how-to-traverse-a-directory-tree-in-python-guide-to-os-walk/).

#### Datetime

A very short example of time manipulation:
```
import datetime

now = datetime.datetime.now()
now.strftime("%Y-%m-%d")
```

See [strftime.org/](http://strftime.org/) for more details about date formatting.