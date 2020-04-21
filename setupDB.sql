create type GreyImage
as char mdarray [ x, y ];

create type RGBCube
as RGBPixel mdarray [ x, y, z ]

create type XGAImage
as RGBPixel mdarray [ x ( 0 : 1023 ), y ( 0 : 767 ) ]