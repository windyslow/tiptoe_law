package protocol

import (
  "os"
  "fmt"
  "time"
  "testing"
  "strconv"
  "runtime/pprof"
)

import (
  "github.com/henrycg/simplepir/pir"
  "github.com/henrycg/simplepir/matrix"
  "github.com/ahenzinger/underhood/underhood"
)

import (
  "github.com/ahenzinger/tiptoe/search/corpus"
  "github.com/ahenzinger/tiptoe/search/utils"
  "github.com/ahenzinger/tiptoe/search/embeddings"
)

const embNumQueries = 20

type testServer struct {
  emb *underhood.Server[matrix.Elem64]
  url *underhood.Server[matrix.Elem32]
}

func setupTestServer(h *TiptoeHint) *testServer {
  out := new(testServer)
  if !h.EmbeddingsHint.IsEmpty() {
    out.emb = underhood.NewServerHintOnly(&h.EmbeddingsHint.Hint)
  }

  if !h.UrlsHint.IsEmpty() {
    out.url = underhood.NewServerHintOnly(&h.UrlsHint.Hint)
  }
  
  return out
}

func applyHint(s *testServer, hq *underhood.HintQuery) *UnderhoodAnswer {
  out := new(UnderhoodAnswer)
  if s.emb != nil {
    out.EmbAnswer = *s.emb.HintAnswer(hq)
  }

  if s.url != nil {
    out.UrlAnswer = *s.url.HintAnswer(hq)
  }

  return out
}

func testRecoverSingle(s *Server, corp *corpus.Corpus) {
  c := NewClient(false /* use coordinator */)

  var h TiptoeHint
  s.GetHint(true, &h)
  c.Setup(&h)
  logHintSize(&h)

  p := h.EmbeddingsHint.Info.P() 

  tserv := setupTestServer(&h)

  for iter := 0; iter < embNumQueries; iter++ {
    f, _ := os.Create("cpu-"+strconv.Itoa(iter)+".prof")
    pprof.StartCPUProfile(f)

    ct := c.PreprocessQuery()

    offlineStart := time.Now()

    uAns := applyHint(tserv, ct)

    logOfflineStats(c.NumDocs(), offlineStart, ct, uAns)
    c.ProcessHintApply(uAns)

    i := utils.RandomIndex(c.NumClusters())
    print("\nc.NumClusters():",c.NumClusters(),"\n")
    emb := readIntegersFromFile("/home/nsklab/yyh/similar/tiptoe/search/mydata/data/q_emb.txt",1)
    query := c.QueryEmbeddings(emb, i)

    start := time.Now()
    var ans pir.Answer[matrix.Elem64]
    s.GetEmbeddingsAnswer(query, &ans)
    logStats(c.NumDocs(), start, query, &ans)
    print("\ni:",i,"\n")
    dec := c.ReconstructEmbeddings(&ans, i)
    clusterIndex := uint64(corp.ClusterToIndex(uint(i)))
    checkAnswer(dec, clusterIndex, p, emb, corp)
    c.checkans(&ans,i,p,corp,emb)
    
    pprof.StopCPUProfile()
  }
}

func testRecoverCluster(s *Server, corp *corpus.Corpus) {
  c := NewClient(false /* use coordinator */)

  var h TiptoeHint
  s.GetHint(true, &h)
  c.Setup(&h)
  logHintSize(&h)

  p := h.EmbeddingsHint.Info.P() 
  tserv := setupTestServer(&h)

  for iter := 0; iter < embNumQueries; iter++ {
    ct := c.PreprocessQuery()

    offlineStart := time.Now()
    uAns := applyHint(tserv, ct)
    logOfflineStats(c.NumDocs(), offlineStart, ct, uAns)
    c.ProcessHintApply(uAns)

    i := utils.RandomIndex(c.NumClusters())
    emb := embeddings.RandomEmbedding(c.params.EmbeddingSlots, (1 << (c.params.SlotBits-1)))
    query := c.QueryEmbeddings(emb, i)

    start := time.Now()
    var ans pir.Answer[matrix.Elem64]
    s.GetEmbeddingsAnswer(query, &ans)
    logStats(c.NumDocs(), start, query, &ans)

    dec := c.ReconstructEmbeddingsWithinCluster(&ans, i)
    checkAnswers(dec, uint(i), p, emb, corp)
  }
}

func testRecoverClusterNetworked(tcp string, useCoordinator bool, corp *corpus.Corpus) {
  c := NewClient(useCoordinator)

  h := c.getHint(false /* keep conn */, tcp)
  c.Setup(h)
  logHintSize(h)

  var tserv *testServer
  p := h.EmbeddingsHint.Info.P() 
  if !useCoordinator {
    tserv = setupTestServer(h)
  }

  for iter := 0; iter < embNumQueries; iter++ {
    ct := c.PreprocessQuery()

    var uAns *UnderhoodAnswer
    offlineStart := time.Now()
    if useCoordinator {
      uAns = c.applyHint(ct, false /* keep conn */, tcp)
    } else {
      uAns = applyHint(tserv, ct)
    }
    logOfflineStats(c.NumDocs(), offlineStart, ct, uAns)
    c.ProcessHintApply(uAns)

    i := utils.RandomIndex(c.NumClusters())
    emb := embeddings.RandomEmbedding(c.params.EmbeddingSlots, (1 << (c.params.SlotBits-1)))
    query := c.QueryEmbeddings(emb, i)

    start := time.Now()
    ans := c.getEmbeddingsAnswer(query, false /* keep conn */, tcp)
    logStats(c.NumDocs(), start, query, ans)

    dec := c.ReconstructEmbeddingsWithinCluster(ans, i)

    checkAnswers(dec, uint(i), p, emb, corp)
  }
}

func testRecoverClusterNetworkedDumpState(tcp string, corp *corpus.Corpus) {
  intermfile := "interm/coordinator_state.log"
  fmt.Println("Dumping coordinator state to file")
  DumpStateToFile(k, intermfile)

  k.hint = nil

  fmt.Println("Loading coordinator state from file")
  LoadStateFromFile(k, intermfile)
  k.SetupConns()

  testRecoverClusterNetworked(tcp, true, corp)

  os.Remove(intermfile)
}

func testEmbeddingsServerDumpState(s *Server, corp *corpus.Corpus) {
  s2.Clear() // needed for the test to pass

  intermfile := "interm/server_state.log"
  DumpStateToFile(s, intermfile)
  fmt.Println("Dumping server state to file")

  LoadStateFromFile(s2, intermfile)
  testRecoverCluster(s2, corp)
  os.Remove(intermfile)
}

func TestEmbeddingsFakeData(t *testing.T) {
  corp := corpus.ReadEmbeddingsCsv(*medcorpus)
  s.PreprocessEmbeddingsFromCorpus(corp, 450 /* hint size in MB */, conf)
  log := conf.EmbeddingServerLog(0)
  print("\nlog:",log)
  DumpStateToFile(s, log)
  k.Setup(1, 0, []string{serverTcp}, false, conf)

  fmt.Printf("Running embedding queries (over %d-doc fake corpus)\n", corp.GetNumDocs())

  testRecoverSingle(s, corp)
  testRecoverCluster(s, corp)
  testEmbeddingsServerDumpState(s, corp)
  testRecoverClusterNetworked(serverTcp, false, corp)
  testRecoverClusterNetworked(coordinatorTcp, true, corp)
  testRecoverClusterNetworkedDumpState(coordinatorTcp, corp)
}

func TestEmbeddingsRealData(t *testing.T) {
  f, _ := os.Create("emb_test.prof")
  pprof.StartCPUProfile(f)
  defer pprof.StopCPUProfile()

  corp := corpus.ReadEmbeddingsTxt(0, 10, conf)
  s.PreprocessEmbeddingsFromCorpus(corp, 450 /* hint size in MB */, conf)
  k.Setup(1, 0, []string{serverTcp}, false, conf)

  fmt.Printf("Running embedding queries (over %d-doc real corpus)\n", corp.GetNumDocs())

  testRecoverSingle(s, corp)
  testRecoverCluster(s, corp)
  testEmbeddingsServerDumpState(s, corp)
  testRecoverClusterNetworked(serverTcp, false, corp)
  testRecoverClusterNetworked(coordinatorTcp, true, corp)
}

func TestEmbeddingsMultipleServersRealData(t *testing.T) {
  numServers := 8
  _, tcps, corp := NewEmbeddingServers(0,
                                       numServers,
				       10,                // clusters per server
                                       30,                // hint sz
				       false,             // log
				       true,              // want corpus
				       true,              // serve
				       conf)

  for ns := 1; ns <= numServers; ns *= 2 {
    c := corpus.Concat(corp[:ns])
    k.Setup(ns, 0, tcps[:ns], false, conf)
    fmt.Printf("Running embedding queries (over %d-doc real corpus with %d servers)\n",
               c.GetNumDocs(), ns)
    testRecoverClusterNetworked(coordinatorTcp, true, c)
  }
}

