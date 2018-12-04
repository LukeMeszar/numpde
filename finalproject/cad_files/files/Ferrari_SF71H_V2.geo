Merge "Ferrari_SF71H_V2.stl";
Physical Surface("Boundary") = {1};
Physical Volume("internal") = {1};
Extrude {0, 0, 10} {
   Surface{6};
   Layers{1};
   Recombine;
}
