import xarray as xr
import arviz as az
import os

print(1e-6*os.path.getsize(
    "Data_and_parameters/PD_sim/PK_comp2/large_data_1pop/pop_V_c_R_0_fixed_K_1_V_1_sigma_PK_inference_obj_wide_V_c_prior.nc"
))
az.rcParams["data.load"] = 'eager'
posterior_samples = az.from_netcdf(
    "Data_and_parameters/PD_sim/PK_comp2/large_data_1pop/pop_V_c_R_0_fixed_K_1_V_1_sigma_PK_inference_obj_wide_V_c_prior.nc"
)
print(posterior_samples.posterior.mean())
az.to_zarr(
    posterior_samples,
    "Data_and_parameters/PD_sim/PK_comp2/large_data_1pop/pop_V_c_R_0_fixed_K_1_V_1_sigma_PK_inference_obj_wide_V_c_prior.zarr",
    # mode='w'
)

print(1e-6*os.path.getsize(
    "Data_and_parameters/PD_sim/PK_comp2/large_data_1pop/pop_V_c_R_0_fixed_K_1_V_1_sigma_PK_inference_obj_wide_V_c_prior.zarr",
))
posterior_samples = az.from_zarr(
    "Data_and_parameters/PD_sim/PK_comp2/large_data_1pop/pop_V_c_R_0_fixed_K_1_V_1_sigma_PK_inference_obj_wide_V_c_prior.zarr",
)
print(posterior_samples.posterior.mean())
