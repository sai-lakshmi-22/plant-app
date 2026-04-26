import h5py
import json
import shutil

src = "model/plant_model.h5"
dst = "model/fixed_plant_model.h5"

# Copy original file
shutil.copy(src, dst)

with h5py.File(dst, "r+") as f:
    config = f.attrs.get("model_config")

    if isinstance(config, bytes):
        config = config.decode("utf-8")

    data = json.loads(config)

    def clean(obj):
        if isinstance(obj, dict):
            obj.pop("quantization_config", None)
            for k, v in list(obj.items()):
                clean(v)
        elif isinstance(obj, list):
            for item in obj:
                clean(item)

    clean(data)

    new_config = json.dumps(data).encode("utf-8")
    f.attrs.modify("model_config", new_config)

print("fixed_plant_model.h5 created")