create type GaussianMeasurement
	as ( mean float, 
		variance float, 
		healpix_coord_variance ulong, 
		time_coord_variance float,
		spectral_coord_variance float)

create type CubeVoxel 
	as GaussianMeasurement mdarray [healPixID, healPixRes, time, spectral]

create type CubeSet
	as set (CubeVoxel null values [-1, 0])

create type CubeSet
	as set (CubeImage null values [-1, -1])
