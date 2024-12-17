int trace_origin_top(const Pythia8::Event& event, int ix, int& bcflag) {
  // see if found W or top
  int id = event[ix].id();
  int ida = abs(id);
  if (ida==24 || ida==6) return id;
  // check b/c origin
  if (bcflag<5 && (ida/100==5 || ida/1000==5)) bcflag = 5;
  else if (bcflag<4 && (ida/100==4 || ida/1000==4)) bcflag = 4;
  // keep digging
  int mother1 = event[ix].mother1();
  int mother2 = event[ix].mother2();
  if (mother1==0) return 0;
  if (mother2==0 || mother2==mother1 || mother2<mother1) return trace_origin_top(event, mother1, bcflag);
  for (int j = mother1; j<=mother2; ++j) {
    // only trace quarks
    int ida = abs(event[j].id());
    if (ida>=1 && ida<=5) {
      int id = trace_origin_top(event, j, bcflag);
      if (abs(id)==24 || abs(id)==6) return id;
    }
  }
  // nothing good
  return 0;
}
