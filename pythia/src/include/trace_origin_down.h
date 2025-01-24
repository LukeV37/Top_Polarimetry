///////////////////////////////////
/// Author: Sasha Khanov 2024   ///
/// Edit: Luke Vaughan Jan 2025 ///
///////////////////////////////////

int trace_origin_down(const Pythia8::Event& event, int ix) {
  // see if down found
  int id = event[ix].id();
  int ida = abs(id);
  if (ida==1 || ida==3) return id; // Check for down type either first gen d or second gen s
  // keep digging
  int mother1 = event[ix].mother1();
  int mother2 = event[ix].mother2();
  if (mother1==0) return 0;
  if (mother2==0 || mother2==mother1 || mother2<mother1) return trace_origin_down(event, mother1);
  for (int j = mother1; j<=mother2; ++j) {
    // only trace quarks
    int ida = abs(event[j].id());
    if (ida>=1 && ida<=5) {
      int id = trace_origin_down(event, j);
      if (ida==1 || ida==3) return id;
    }
  }
  // nothing good
  return 0;
}
