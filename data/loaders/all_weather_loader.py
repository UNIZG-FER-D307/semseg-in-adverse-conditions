from .dark_loader import prepare_dark_datasets
from .snow_loader import prepare_snow_datasets
from .rainy_loader import prepare_rainy_datasets
from .foggy_loader import prepare_foggy_datasets
from .robust_loader import prepare_robust_datasets
from .city_loader import prepare_city_datasets
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from data.datasets import UniformClassDataset


def _get_ds_len(ds):
    return sum([len(d) for d in ds])


def load_all_weather_train(dataroots, bs, train_transforms, config):
    rain_ds = prepare_rainy_datasets(dataroots, train_transforms, config)
    fog_ds = prepare_foggy_datasets(dataroots, train_transforms, config)
    snow_ds = prepare_snow_datasets(dataroots, train_transforms, config)
    dark_ds = prepare_dark_datasets(dataroots, train_transforms, config)
    robust_ds = prepare_robust_datasets(dataroots, train_transforms, config)
    city_ds = prepare_city_datasets(dataroots, train_transforms, config)

    print(f"Rain images {_get_ds_len(rain_ds)}")
    print(f"Fog images {_get_ds_len(fog_ds)}")
    print(f"Snow images {_get_ds_len(snow_ds)}")
    print(f"Dark images {_get_ds_len(dark_ds)}")
    print(f"Robust images {_get_ds_len(robust_ds)}")
    print(f"City images {_get_ds_len(city_ds)}")

    all_conditions = rain_ds + fog_ds + snow_ds + dark_ds + robust_ds + city_ds
    if config["uniform"]:
        all_conditions = [
            UniformClassDataset(ds, class_uniform_pct=0.75, class_uniform_tile=512)
            for ds in all_conditions
        ]
        print("> Created uniform dataset")
    ds = ConcatDataset(all_conditions)

    print(f"> Loaded {len(ds)} train images.")
    train_loader = DataLoader(
        ds, batch_size=bs, shuffle=True, pin_memory=True, num_workers=5, drop_last=True
    )
    return train_loader


def load_acdc_weathers_train(dataroots, bs, train_transforms, config):
    rain_ds = prepare_rainy_datasets(dataroots, train_transforms, config)
    fog_ds = prepare_foggy_datasets(dataroots, train_transforms, config)
    snow_ds = prepare_snow_datasets(dataroots, train_transforms, config)
    dark_ds = prepare_dark_datasets(dataroots, train_transforms, config)

    print(f"Rain images {_get_ds_len(rain_ds)}")
    print(f"Fog images {_get_ds_len(fog_ds)}")
    print(f"Snow images {_get_ds_len(snow_ds)}")
    print(f"Dark images {_get_ds_len(dark_ds)}")

    all_conditions = rain_ds + fog_ds + snow_ds + dark_ds
    ds = ConcatDataset(all_conditions)

    print(f"> Loaded {len(ds)} train images.")
    train_loader = DataLoader(
        ds, batch_size=bs, shuffle=True, pin_memory=True, num_workers=5, drop_last=True
    )
    return train_loader
