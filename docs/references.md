Bibliography of the crate.

Every entry has an HTML anchor named after its key in lower case, so the rustdoc of any item cites
it as `[\[Kah62\]](crate::references#kah62)`. Keys are formed from the authors' initials (or the
first author's name) and the two-digit year. The entries are grouped by algorithm family, in the
order of the crate documentation; within a group the cited classics come first, then the
background and the deferred work.

**Verification.** Every DOI below was resolved against the Crossref REST API
(`api.crossref.org/works/<doi>`) or, for DataCite DOIs (JALC, LIPIcs, OpenProceedings), against
`doi.org` content negotiation, in October 2026; authors, title, venue, volume, issue, pages and
year agree with the registry record. Entries without a DOI were checked by web search against the
publisher, the ACL Anthology, arXiv, DBLP or a library catalogue, as noted in each entry.
Qualifiers such as "background only" say how the crate relies on a source.

# Topological order, dynamic programming and semirings

- <a id="kah62"></a>**\[Kah62\]** A. B. Kahn. Topological sorting of large networks.
  *Communications of the ACM* 5(11):558–562, 1962.
  [doi:10.1145/368996.369025](https://doi.org/10.1145/368996.369025).
- <a id="bel58"></a>**\[Bel58\]** R. Bellman. On a routing problem. *Quarterly of Applied
  Mathematics* 16(1):87–90, 1958. [doi:10.1090/qam/102435](https://doi.org/10.1090/qam/102435).
- <a id="clrs22"></a>**\[CLRS22\]** T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein.
  *Introduction to Algorithms*, 4th ed. MIT Press, 2022. ISBN 978-0-262-04630-5. Sec. 20.4
  (topological sort by depth-first search) and Sec. 22.2 "Single-source shortest paths in directed
  acyclic graphs" (Sec. 22.4 and 24.2 in the 3rd ed.). No DOI. The section numbers were checked
  against a copy of the 4th-edition table of contents found by web search, not against the
  publisher's page.
- <a id="kw59"></a>**\[KW59\]** J. E. Kelley Jr., M. R. Walker. Critical-path planning and
  scheduling. *Papers presented at the Eastern Joint IRE-AIEE-ACM Computer Conference*, Boston,
  December 1–3, 1959, pp. 160–173.
  [doi:10.1145/1460299.1460318](https://doi.org/10.1145/1460299.1460318).
- <a id="mrcf59"></a>**\[MRCF59\]** D. G. Malcolm, J. H. Roseboom, C. E. Clark, W. Fazar.
  Application of a technique for research and development program evaluation. *Operations
  Research* 7(5):646–669, 1959. [doi:10.1287/opre.7.5.646](https://doi.org/10.1287/opre.7.5.646).
- <a id="vit67"></a>**\[Vit67\]** A. J. Viterbi. Error bounds for convolutional codes and an
  asymptotically optimum decoding algorithm. *IEEE Transactions on Information Theory*
  13(2):260–269, 1967. [doi:10.1109/TIT.1967.1054010](https://doi.org/10.1109/TIT.1967.1054010).
- <a id="leh77"></a>**\[Leh77\]** D. J. Lehmann. Algebraic structures for transitive closure.
  *Theoretical Computer Science* 4(1):59–76, 1977.
  [doi:10.1016/0304-3975(77)90056-1](https://doi.org/10.1016/0304-3975(77)90056-1).
- <a id="goo99"></a>**\[Goo99\]** J. Goodman. Semiring parsing. *Computational Linguistics*
  25(4):573–606, 1999. ACL Anthology [J99-4004](https://aclanthology.org/J99-4004/); no DOI is
  registered (the anthology page and its BibTeX give pp. 573–606).
- <a id="moh02"></a>**\[Moh02\]** M. Mohri. Semiring frameworks and algorithms for
  shortest-distance problems. *Journal of Automata, Languages and Combinatorics* 7(3):321–350,
  2002. [doi:10.25596/jalc-2002-321](https://doi.org/10.25596/jalc-2002-321).
- <a id="gm08"></a>**\[GM08\]** M. Gondran, M. Minoux. *Graphs, Dioids and Semirings: New Models
  and Algorithms*. Operations Research/Computer Science Interfaces 41, Springer, 2008.
  [doi:10.1007/978-0-387-75450-5](https://doi.org/10.1007/978-0-387-75450-5).
- <a id="hua08"></a>**\[Hua08\]** L. Huang. Advanced dynamic programming in semiring and
  hypergraph frameworks. *Coling 2008: Advanced Dynamic Programming in Computational Linguistics:
  Theory, Algorithms and Applications — Tutorial notes*, Manchester, 2008, pp. 1–18.
  ACL Anthology [C08-5001](https://aclanthology.org/C08-5001/); no DOI.
- <a id="eis02"></a>**\[Eis02\]** J. Eisner. Parameter estimation for probabilistic finite-state
  transducers. *Proc. 40th Annual Meeting of the ACL (ACL 2002)*, pp. 1–8.
  [doi:10.3115/1073083.1073085](https://doi.org/10.3115/1073083.1073085). The expectation
  semiring; background only (the instance is not implemented).
- <a id="le09"></a>**\[LE09\]** Z. Li, J. Eisner. First- and second-order expectation semirings
  with applications to minimum-risk training on translation forests. *Proc. EMNLP 2009*,
  pp. 40–51. [doi:10.3115/1699510.1699517](https://doi.org/10.3115/1699510.1699517). Background
  only.
- <a id="gkt07"></a>**\[GKT07\]** T. J. Green, G. Karvounarakis, V. Tannen. Provenance semirings.
  *Proc. PODS 2007*, pp. 31–40.
  [doi:10.1145/1265530.1265535](https://doi.org/10.1145/1265530.1265535). Background for the
  provenance reading of the `prob` semiring with `inv-outdeg` weights.
- <a id="hig93"></a>**\[Hig93\]** N. J. Higham. The accuracy of floating point summation. *SIAM
  Journal on Scientific Computing* 14(4):783–799, 1993.
  [doi:10.1137/0914050](https://doi.org/10.1137/0914050). Pairwise-summation error bounds used by
  the certification tolerances and the path-count totals.
- <a id="kah65"></a>**\[Kah65\]** W. Kahan. Pracniques: further remarks on reducing truncation
  errors. *Communications of the ACM* 8(1):40, 1965.
  [doi:10.1145/363707.363723](https://doi.org/10.1145/363707.363723). Compensated summation;
  background only.
- <a id="bv04"></a>**\[BV04\]** P. Boldi, S. Vigna. The WebGraph framework I: compression
  techniques. *Proc. 13th International World Wide Web Conference (WWW 2004)*, pp. 595–602.
  [doi:10.1145/988672.988752](https://doi.org/10.1145/988672.988752). The BV format of every input
  graph (read with the Rust port `webgraph-rs`).

Parallel evaluation (deferred work, see the README):

- <a id="as89"></a>**\[AS89\]** E. Anderson, Y. Saad. Solving sparse triangular linear systems on
  parallel computers. *International Journal of High Speed Computing* 1(1):73–95, 1989.
  [doi:10.1142/S0129053389000056](https://doi.org/10.1142/S0129053389000056). Level scheduling.
- <a id="cyt86"></a>**\[Cyt86\]** R. Cytron. Doacross: beyond vectorization for multiprocessors.
  *Proc. International Conference on Parallel Processing (ICPP 1986)*, pp. 836–844. No DOI
  (verified via DBLP and the IBM Research publication page).
- <a id="llhdv16"></a>**\[LLHDV16\]** W. Liu, A. Li, J. Hogg, I. S. Duff, B. Vinter. A
  synchronization-free algorithm for parallel sparse triangular solves. *Euro-Par 2016*, LNCS
  9833, pp. 617–630. [doi:10.1007/978-3-319-43659-3_45](https://doi.org/10.1007/978-3-319-43659-3_45).
- <a id="lf80"></a>**\[LF80\]** R. E. Ladner, M. J. Fischer. Parallel prefix computation.
  *Journal of the ACM* 27(4):831–838, 1980.
  [doi:10.1145/322217.322232](https://doi.org/10.1145/322217.322232).
- <a id="mr85"></a>**\[MR85\]** G. L. Miller, J. H. Reif. Parallel tree contraction and its
  application. *Proc. 26th Annual Symposium on Foundations of Computer Science (FOCS 1985)*,
  pp. 478–489. [doi:10.1109/SFCS.1985.43](https://doi.org/10.1109/SFCS.1985.43).

# Dominators and nearest common ancestors

- <a id="lt79"></a>**\[LT79\]** T. Lengauer, R. E. Tarjan. A fast algorithm for finding dominators
  in a flowgraph. *ACM Transactions on Programming Languages and Systems* 1(1):121–141, 1979.
  [doi:10.1145/357062.357071](https://doi.org/10.1145/357062.357071).
- <a id="chk01"></a>**\[CHK01\]** K. D. Cooper, T. J. Harvey, K. Kennedy. A simple, fast dominance
  algorithm. Rice University, Department of Computer Science, Technical Report TR-06-33870
  (also cited as TR-06-38870). Usually cited as *Software Practice & Experience* 4:1–10, 2001, but
  Crossref has no journal record and there is no DOI, so it is cited as the Rice report (verified by
  web search: the report and the authors' copies are online).
- <a id="ahlt99"></a>**\[AHLT99\]** S. Alstrup, D. Harel, P. W. Lauridsen, M. Thorup. Dominators in
  linear time. *SIAM Journal on Computing* 28(6):2117–2132, 1999.
  [doi:10.1137/S0097539797317263](https://doi.org/10.1137/S0097539797317263).
- <a id="bgkrtw08"></a>**\[BGKRTW08\]** A. L. Buchsbaum, L. Georgiadis, H. Kaplan, A. Rogers,
  R. E. Tarjan, J. R. Westbrook. Linear-time algorithms for dominators and other path-evaluation
  problems. *SIAM Journal on Computing* 38(4):1533–1573, 2008.
  [doi:10.1137/070693217](https://doi.org/10.1137/070693217). Preprint arXiv:cs/0207061.
- <a id="gt15"></a>**\[GT15\]** L. Georgiadis, R. E. Tarjan. Dominator tree certification and
  divergent spanning trees. *ACM Transactions on Algorithms* 12(1):11:1–11:42, 2015 (issue dated
  2016). [doi:10.1145/2764913](https://doi.org/10.1145/2764913). Addendum: *ACM TALG* 12(4):56,
  2016, [doi:10.1145/2928271](https://doi.org/10.1145/2928271). Cited for low-high orders and
  derived arcs; its lemma numbering is not relied upon.
- <a id="gtw06"></a>**\[GTW06\]** L. Georgiadis, R. E. Tarjan, R. F. Werneck. Finding dominators in
  practice. *Journal of Graph Algorithms and Applications* 10(1):69–94, 2006.
  [doi:10.7155/jgaa.00119](https://doi.org/10.7155/jgaa.00119).
- <a id="och83"></a>**\[Och83\]** R. Ochranová. Finding dominators. *Foundations of Computation
  Theory (FCT 1983)*, LNCS 158, pp. 328–334.
  [doi:10.1007/3-540-12689-9_115](https://doi.org/10.1007/3-540-12689-9_115). Commonly cited for
  dominators in acyclic graphs; not read, cited as background only.
- <a id="rr94"></a>**\[RR94\]** G. Ramalingam, T. Reps. An incremental algorithm for maintaining the
  dominator tree of a reducible flowgraph. *Proc. 21st ACM POPL (POPL 1994)*, pp. 287–296.
  [doi:10.1145/174675.177905](https://doi.org/10.1145/174675.177905). The acyclic-dominators-by-NCA
  algorithm.
- <a id="mye83"></a>**\[Mye83\]** E. W. Myers. An applicative random-access stack. *Information
  Processing Letters* 17(5):241–248, 1983.
  [doi:10.1016/0020-0190(83)90106-0](https://doi.org/10.1016/0020-0190(83)90106-0). Skew-binary
  jump pointers.
- <a id="bf00"></a>**\[BF00\]** M. A. Bender, M. Farach-Colton. The LCA problem revisited.
  *LATIN 2000*, LNCS 1776, pp. 88–94.
  [doi:10.1007/10719839_9](https://doi.org/10.1007/10719839_9).
- <a id="bfpss05"></a>**\[BFPSS05\]** M. A. Bender, M. Farach-Colton, G. Pemmasani, S. Skiena,
  P. Sumazin. Lowest common ancestors in trees and directed acyclic graphs. *Journal of
  Algorithms* 57(2):75–94, 2005.
  [doi:10.1016/j.jalgor.2005.08.001](https://doi.org/10.1016/j.jalgor.2005.08.001). DAG LCA sets
  (deferred).
- <a id="lm69"></a>**\[LM69\]** E. S. Lowry, C. W. Medlock. Object code optimization.
  *Communications of the ACM* 12(1):13–22, 1969.
  [doi:10.1145/362835.362838](https://doi.org/10.1145/362835.362838). Origin of the term
  "dominator".
- <a id="pro59"></a>**\[Pro59\]** R. T. Prosser. Applications of Boolean matrices to the analysis of
  flow diagrams. *Papers presented at the Eastern Joint IRE-AIEE-ACM Computer Conference*, 1959,
  pp. 133–138. [doi:10.1145/1460299.1460314](https://doi.org/10.1145/1460299.1460314).
- <a id="all70"></a>**\[All70\]** F. E. Allen. Control flow analysis. *ACM SIGPLAN Notices*
  5(7):1–19, 1970. [doi:10.1145/390013.808479](https://doi.org/10.1145/390013.808479).
- <a id="cfrwz91"></a>**\[CFRWZ91\]** R. Cytron, J. Ferrante, B. K. Rosen, M. N. Wegman,
  F. K. Zadeck. Efficiently computing static single assignment form and the control dependence
  graph. *ACM Transactions on Programming Languages and Systems* 13(4):451–490, 1991.
  [doi:10.1145/115372.115320](https://doi.org/10.1145/115372.115320). Post-dominators and control
  dependence.
- <a id="mit06"></a>**\[Mit06\]** N. Mitchell. The runtime structure of object ownership. *ECOOP
  2006*, LNCS 4067, pp. 74–98.
  [doi:10.1007/11785477_5](https://doi.org/10.1007/11785477_5). Dominator-tree retained size.
- <a id="ab04"></a>**\[AB04\]** S. Allesina, A. Bodini. Who dominates whom in the ecosystem? Energy
  flow bottlenecks and cascading extinctions. *Journal of Theoretical Biology* 230(3):351–358,
  2004. [doi:10.1016/j.jtbi.2004.05.009](https://doi.org/10.1016/j.jtbi.2004.05.009).

Background on dominators (cited by the design notes, not by the code):

- <a id="tar74"></a>**\[Tar74\]** R. E. Tarjan. Finding dominators in directed graphs. *SIAM
  Journal on Computing* 3(1):62–89, 1974. [doi:10.1137/0203006](https://doi.org/10.1137/0203006).
- <a id="pm72"></a>**\[PM72\]** P. W. Purdom, E. F. Moore. Algorithm 430: Immediate predominators
  in a directed graph. *Communications of the ACM* 15(8):777–778, 1972.
  [doi:10.1145/361532.361566](https://doi.org/10.1145/361532.361566).
- <a id="har85"></a>**\[Har85\]** D. Harel. A linear algorithm for finding dominators in flow graphs
  and related problems. *Proc. 17th ACM STOC (STOC 1985)*, pp. 185–194.
  [doi:10.1145/22145.22166](https://doi.org/10.1145/22145.22166). The claim was incomplete; it was
  completed by [\[AHLT99\]](#ahlt99).
- <a id="ram02"></a>**\[Ram02\]** G. Ramalingam. On loops, dominators, and dominance frontiers.
  *ACM Transactions on Programming Languages and Systems* 24(5):455–490, 2002.
  [doi:10.1145/570886.570887](https://doi.org/10.1145/570886.570887). Reduction from general to
  acyclic dominators.
- <a id="gt05"></a>**\[GT05\]** L. Georgiadis, R. E. Tarjan. Dominator tree verification and
  vertex-disjoint paths. *Proc. 16th ACM-SIAM SODA (SODA 2005)*, pp. 433–442. No DOI (verified via
  the authors' publication pages).
- <a id="gt12"></a>**\[GT12\]** L. Georgiadis, R. E. Tarjan. Dominators, directed bipolar orders,
  and independent spanning trees. *ICALP 2012*, LNCS 7391, pp. 375–386.
  [doi:10.1007/978-3-642-31594-7_32](https://doi.org/10.1007/978-3-642-31594-7_32).
- <a id="glpt13"></a>**\[GLPT13\]** L. Georgiadis, L. Laura, N. Parotsidis, R. E. Tarjan. Dominator
  certification and independent spanning trees: an experimental study. *SEA 2013*, LNCS 7933,
  pp. 284–295. [doi:10.1007/978-3-642-38527-8_26](https://doi.org/10.1007/978-3-642-38527-8_26).
- <a id="gp13"></a>**\[GP13\]** L. Georgiadis, N. Parotsidis. Dominators in directed graphs: a
  survey of recent results, applications, and open problems. *Proc. 2nd International Symposium
  on Computing in Informatics and Mathematics (ISCIM 2013)*, pp. 15–20. No DOI; the venue and pages
  were confirmed on the first author's publication list, the title only by the design notes.
- <a id="fgmt13"></a>**\[FGMT13\]** W. Fraczak, L. Georgiadis, A. Miller, R. E. Tarjan. Finding
  dominators via disjoint set union. *Journal of Discrete Algorithms* 23:2–20, 2013.
  [doi:10.1016/j.jda.2013.10.003](https://doi.org/10.1016/j.jda.2013.10.003). Corrections: *JDA*
  26:106–110, 2014, [doi:10.1016/j.jda.2014.01.002](https://doi.org/10.1016/j.jda.2014.01.002).
- <a id="pgr87"></a>**\[PGR87\]** S. R. Pawagi, P. S. Gopalakrishnan, I. V. Ramakrishnan. Computing
  dominators in parallel. *Information Processing Letters* 24(4):217–221, 1987.
  [doi:10.1016/0020-0190(87)90136-0](https://doi.org/10.1016/0020-0190(87)90136-0).
- <a id="jpp94"></a>**\[JPP94\]** R. Johnson, D. Pearson, K. Pingali. The program structure tree:
  computing control regions in linear time. *Proc. PLDI 1994*, *ACM SIGPLAN Notices*
  29(6):171–185. [doi:10.1145/773473.178258](https://doi.org/10.1145/773473.178258).
- <a id="ils12"></a>**\[ILS12\]** G. F. Italiano, L. Laura, F. Santaroni. Finding strong bridges and
  strong articulation points in linear time. *Theoretical Computer Science* 447:74–84, 2012.
  [doi:10.1016/j.tcs.2011.11.011](https://doi.org/10.1016/j.tcs.2011.11.011).
- <a id="gab90"></a>**\[Gab90\]** H. N. Gabow. Data structures for weighted matching and nearest
  common ancestors with linking. *Proc. 1st ACM-SIAM SODA (SODA 1990)*, pp. 434–443 (no DOI).
  Journal version: A data structure for nearest common ancestors with linking. *ACM Transactions on
  Algorithms* 13(4):45, 2017. [doi:10.1145/3108240](https://doi.org/10.1145/3108240).
- <a id="at00"></a>**\[AT00\]** S. Alstrup, M. Thorup. Optimal pointer algorithms for finding
  nearest common ancestors in dynamic trees. *Journal of Algorithms* 35(2):169–188, 2000.
  [doi:10.1006/jagm.2000.1079](https://doi.org/10.1006/jagm.2000.1079).
- <a id="bf04"></a>**\[BF04\]** M. A. Bender, M. Farach-Colton. The level ancestor problem
  simplified. *Theoretical Computer Science* 321(1):5–12, 2004.
  [doi:10.1016/j.tcs.2003.05.002](https://doi.org/10.1016/j.tcs.2003.05.002).
- <a id="bv94"></a>**\[BV94\]** O. Berkman, U. Vishkin. Finding level-ancestors in trees. *Journal
  of Computer and System Sciences* 48(2):214–230, 1994.
  [doi:10.1016/S0022-0000(05)80002-9](https://doi.org/10.1016/S0022-0000(05)80002-9).
- <a id="ht84"></a>**\[HT84\]** D. Harel, R. E. Tarjan. Fast algorithms for finding nearest common
  ancestors. *SIAM Journal on Computing* 13(2):338–355, 1984.
  [doi:10.1137/0213024](https://doi.org/10.1137/0213024).
- <a id="ds87"></a>**\[DS87\]** P. Dietz, D. Sleator. Two algorithms for maintaining order in a
  list. *Proc. 19th ACM STOC (STOC 1987)*, pp. 365–372.
  [doi:10.1145/28395.28434](https://doi.org/10.1145/28395.28434).
- <a id="bcdfz02"></a>**\[BCDFZ02\]** M. A. Bender, R. Cole, E. D. Demaine, M. Farach-Colton,
  J. Zito. Two simplified algorithms for maintaining order in a list. *ESA 2002*, LNCS 2461,
  pp. 152–164. [doi:10.1007/3-540-45749-6_17](https://doi.org/10.1007/3-540-45749-6_17).

# Reachability

- <a id="ffgm07"></a>**\[FFGM07\]** P. Flajolet, É. Fusy, O. Gandouet, F. Meunier. HyperLogLog: the
  analysis of a near-optimal cardinality estimation algorithm. *Proc. 2007 Conference on Analysis
  of Algorithms (AofA 2007)*, DMTCS Proceedings vol. AH, pp. 127–146.
  [doi:10.46298/dmtcs.3545](https://doi.org/10.46298/dmtcs.3545).
- <a id="qkt16"></a>**\[QKT16\]** J. Qin, D. Kim, Y. Tung. LogLog-Beta and more: a new algorithm for
  cardinality estimation based on LogLog counting. arXiv:1612.02284, 2016 (v3, 2020).
  [arXiv:1612.02284](https://arxiv.org/abs/1612.02284); no DOI. The default estimator of
  `card-est-array`.
- <a id="brv11"></a>**\[BRV11\]** P. Boldi, M. Rosa, S. Vigna. HyperANF: approximating the
  neighbourhood function of very large graphs on a budget. *Proc. 20th International World Wide
  Web Conference (WWW 2011)*, pp. 625–634.
  [doi:10.1145/1963405.1963493](https://doi.org/10.1145/1963405.1963493).
- <a id="bv13"></a>**\[BV13\]** P. Boldi, S. Vigna. In-core computation of geometric centralities
  with HyperBall: a hundred billion nodes and beyond. *Proc. 2013 IEEE 13th International
  Conference on Data Mining Workshops (ICDMW 2013)*, pp. 621–628.
  [doi:10.1109/ICDMW.2013.10](https://doi.org/10.1109/ICDMW.2013.10).
- <a id="coh97"></a>**\[Coh97\]** E. Cohen. Size-estimation framework with applications to
  transitive closure and reachability. *Journal of Computer and System Sciences* 55(3):441–453,
  1997. [doi:10.1006/jcss.1997.1534](https://doi.org/10.1006/jcss.1997.1534). Conference version:
  Estimating the size of the transitive closure in linear time, *FOCS 1994*, pp. 190–200,
  [doi:10.1109/SFCS.1994.365694](https://doi.org/10.1109/SFCS.1994.365694).
- <a id="hnh13"></a>**\[HNH13\]** S. Heule, M. Nunkesser, A. Hall. HyperLogLog in practice:
  algorithmic engineering of a state of the art cardinality estimation algorithm. *Proc. 16th
  International Conference on Extending Database Technology (EDBT 2013)*, pp. 683–692.
  [doi:10.1145/2452376.2452456](https://doi.org/10.1145/2452376.2452456).
- <a id="abj89"></a>**\[ABJ89\]** R. Agrawal, A. Borgida, H. V. Jagadish. Efficient management of
  transitive relationships in large data and knowledge bases. *Proc. ACM SIGMOD 1989*,
  pp. 253–262. [doi:10.1145/67544.66950](https://doi.org/10.1145/67544.66950).
- <a id="jag90"></a>**\[Jag90\]** H. V. Jagadish. A compression technique to materialize transitive
  closure. *ACM Transactions on Database Systems* 15(4):558–598, 1990.
  [doi:10.1145/99935.99944](https://doi.org/10.1145/99935.99944). Chain decompositions for
  reachability labels.
- <a id="slf14"></a>**\[SLF14\]** G. L. Steele Jr., D. Lea, C. H. Flood. Fast splittable pseudorandom
  number generators. *Proc. 2014 ACM OOPSLA*, pp. 453–472, 2014.
  [doi:10.1145/2660193.2660195](https://doi.org/10.1145/2660193.2660195). SplitMix64, the seeded
  hash of `reach::hash`.
- <a id="sim88"></a>**\[Sim88\]** K. Simon. An improved algorithm for transitive closure on acyclic
  digraphs. *Theoretical Computer Science* 58(1–3):325–346, 1988.
  [doi:10.1016/0304-3975(88)90032-1](https://doi.org/10.1016/0304-3975(88)90032-1). Conference
  version: *ICALP 1986*, LNCS 226, pp. 376–386,
  [doi:10.1007/3-540-16761-7_87](https://doi.org/10.1007/3-540-16761-7_87).
- <a id="chkz03"></a>**\[CHKZ03\]** E. Cohen, E. Halperin, H. Kaplan, U. Zwick. Reachability and
  distance queries via 2-hop labels. *SIAM Journal on Computing* 32(5):1338–1355, 2003.
  [doi:10.1137/S0097539702403098](https://doi.org/10.1137/S0097539702403098).
- <a id="ycz10"></a>**\[YCZ10\]** H. Yıldırım, V. Chaoji, M. J. Zaki. GRAIL: scalable reachability
  index for large graphs. *Proceedings of the VLDB Endowment* 3(1–2):276–284, 2010.
  [doi:10.14778/1920841.1920879](https://doi.org/10.14778/1920841.1920879). Journal version: GRAIL:
  a scalable index for reachability queries in very large graphs, *The VLDB Journal*
  21(4):509–534, 2012, [doi:10.1007/s00778-011-0256-4](https://doi.org/10.1007/s00778-011-0256-4).
- <a id="sabw13"></a>**\[SABW13\]** S. Seufert, A. Anand, S. Bedathur, G. Weikum. FERRARI: flexible
  and efficient reachability range assignment for graph indexing. *Proc. 29th IEEE International
  Conference on Data Engineering (ICDE 2013)*, pp. 1009–1020.
  [doi:10.1109/ICDE.2013.6544893](https://doi.org/10.1109/ICDE.2013.6544893).
- <a id="wylj14"></a>**\[WYLJ14\]** H. Wei, J. X. Yu, C. Lu, R. Jin. Reachability querying: an
  independent permutation labeling approach. *Proceedings of the VLDB Endowment* 7(12):1191–1202,
  2014. [doi:10.14778/2732977.2732992](https://doi.org/10.14778/2732977.2732992). Journal version:
  *The VLDB Journal* 27(1):1–26, 2018,
  [doi:10.1007/s00778-017-0468-3](https://doi.org/10.1007/s00778-017-0468-3).
- <a id="szwy17"></a>**\[SZWY17\]** J. Su, Q. Zhu, H. Wei, J. X. Yu. Reachability querying: can it be
  even faster? *IEEE Transactions on Knowledge and Data Engineering* 29(3):683–697, 2017.
  [doi:10.1109/TKDE.2016.2631160](https://doi.org/10.1109/TKDE.2016.2631160).
- <a id="hst22"></a>**\[HST22\]** K. Hanauer, C. Schulz, J. Trummer. O'Reach: even faster
  reachability in large graphs. *ACM Journal of Experimental Algorithmics* 27:1–27, 2022.
  [doi:10.1145/3556540](https://doi.org/10.1145/3556540). Conference version: *SEA 2021*, LIPIcs
  190, 13:1–13:24, [doi:10.4230/LIPIcs.SEA.2021.13](https://doi.org/10.4230/LIPIcs.SEA.2021.13);
  preprint arXiv:2008.10932.
- <a id="zbo23"></a>**\[ZBO23\]** C. Zhang, A. Bonifati, M. T. Özsu. An overview of reachability
  indexes on graphs. *Companion of the 2023 International Conference on Management of Data
  (SIGMOD-Companion 2023)*, pp. 61–68.
  [doi:10.1145/3555041.3589408](https://doi.org/10.1145/3555041.3589408).

Background on sketches and reachability indexes (cited by the design notes, not by the code):

- <a id="ck07"></a>**\[CK07\]** E. Cohen, H. Kaplan. Summarizing data using bottom-k sketches.
  *Proc. 26th ACM PODC (PODC 2007)*, pp. 225–234.
  [doi:10.1145/1281100.1281133](https://doi.org/10.1145/1281100.1281133).
- <a id="bcfm00"></a>**\[BCFM00\]** A. Z. Broder, M. Charikar, A. M. Frieze, M. Mitzenmacher.
  Min-wise independent permutations. *Journal of Computer and System Sciences* 60(3):630–659, 2000.
  [doi:10.1006/jcss.1999.1690](https://doi.org/10.1006/jcss.1999.1690).
- <a id="pgf02"></a>**\[PGF02\]** C. R. Palmer, P. B. Gibbons, C. Faloutsos. ANF: a fast and scalable
  tool for data mining in massive graphs. *Proc. 8th ACM SIGKDD (KDD 2002)*, pp. 81–90.
  [doi:10.1145/775047.775059](https://doi.org/10.1145/775047.775059).
- <a id="coh15"></a>**\[Coh15\]** E. Cohen. All-distances sketches, revisited: HIP estimators for
  massive graphs analysis. *IEEE Transactions on Knowledge and Data Engineering* 27(9):2320–2334,
  2015. [doi:10.1109/TKDE.2015.2411606](https://doi.org/10.1109/TKDE.2015.2411606).
- <a id="yaiy13"></a>**\[YAIY13\]** Y. Yano, T. Akiba, Y. Iwata, Y. Yoshida. Fast and scalable
  reachability queries on graphs by pruned labeling with landmarks and paths. *Proc. 22nd ACM CIKM
  (CIKM 2013)*, pp. 1601–1606.
  [doi:10.1145/2505515.2505724](https://doi.org/10.1145/2505515.2505724).
- <a id="chwf13"></a>**\[CHWF13\]** J. Cheng, S. Huang, H. Wu, A. W.-C. Fu. TF-Label: a
  topological-folding labeling scheme for reachability querying in a large graph. *Proc. ACM
  SIGMOD 2013*, pp. 193–204.
  [doi:10.1145/2463676.2465286](https://doi.org/10.1145/2463676.2465286).
- <a id="vcmz14"></a>**\[VCMZ14\]** R. R. Veloso, L. Cerf, W. Meira Jr., M. J. Zaki. Reachability
  queries in very large graphs: a fast refined online search approach. *Proc. 17th International
  Conference on Extending Database Technology (EDBT 2014)*, pp. 511–522.
  [doi:10.5441/002/edbt.2014.46](https://doi.org/10.5441/002/edbt.2014.46).
- <a id="zlwx14"></a>**\[ZLWX14\]** A. D. Zhu, W. Lin, S. Wang, X. Xiao. Reachability queries on
  large dynamic graphs: a total order approach. *Proc. ACM SIGMOD 2014*, pp. 1323–1334.
  [doi:10.1145/2588555.2612181](https://doi.org/10.1145/2588555.2612181).
- <a id="jxrf09"></a>**\[JXRF09\]** R. Jin, Y. Xiang, N. Ruan, D. Fuhry. 3-HOP: a high-compression
  indexing scheme for reachability query. *Proc. ACM SIGMOD 2009*, pp. 813–826.
  [doi:10.1145/1559845.1559930](https://doi.org/10.1145/1559845.1559930).
- <a id="jrxw11"></a>**\[JRXW11\]** R. Jin, N. Ruan, Y. Xiang, H. Wang. Path-tree: an efficient
  reachability indexing scheme for large directed graphs. *ACM Transactions on Database Systems*
  36(1):7, 2011. [doi:10.1145/1929934.1929941](https://doi.org/10.1145/1929934.1929941).

# Transitive reduction, path covers and width

- <a id="agu72"></a>**\[AGU72\]** A. V. Aho, M. R. Garey, J. D. Ullman. The transitive reduction of
  a directed graph. *SIAM Journal on Computing* 1(2):131–137, 1972.
  [doi:10.1137/0201008](https://doi.org/10.1137/0201008).
- <a id="dil50"></a>**\[Dil50\]** R. P. Dilworth. A decomposition theorem for partially ordered
  sets. *Annals of Mathematics* 51(1):161–166, 1950.
  [doi:10.2307/1969503](https://doi.org/10.2307/1969503).
- <a id="ful56"></a>**\[Ful56\]** D. R. Fulkerson. Note on Dilworth's decomposition theorem for
  partially ordered sets. *Proceedings of the American Mathematical Society* 7(4):701–702, 1956.
  [doi:10.1090/S0002-9939-1956-0078334-6](https://doi.org/10.1090/S0002-9939-1956-0078334-6).
- <a id="kon31"></a>**\[Kon31\]** D. Kőnig. Gráfok és mátrixok. *Matematikai és Fizikai Lapok*
  38:116–119, 1931. No DOI (verified via web search; English translation by G. Szárnyas,
  "Graphs and matrices", arXiv:2009.03780).
- <a id="hk73"></a>**\[HK73\]** J. E. Hopcroft, R. M. Karp. An n^{5/2} algorithm for maximum
  matchings in bipartite graphs. *SIAM Journal on Computing* 2(4):225–231, 1973.
  [doi:10.1137/0202019](https://doi.org/10.1137/0202019).
- <a id="bg77"></a>**\[BG77\]** F. T. Boesch, J. F. Gimpel. Covering points of a digraph with
  point-disjoint paths and its application to code optimization. *Journal of the ACM*
  24(2):192–198, 1977. [doi:10.1145/322003.322005](https://doi.org/10.1145/322003.322005).
- <a id="nh79"></a>**\[NH79\]** S. C. Ntafos, S. L. Hakimi. On path cover problems in digraphs and
  applications to program testing. *IEEE Transactions on Software Engineering* SE-5(5):520–529,
  1979. [doi:10.1109/TSE.1979.234213](https://doi.org/10.1109/TSE.1979.234213).
- <a id="ff56"></a>**\[FF56\]** L. R. Ford Jr., D. R. Fulkerson. Maximal flow through a network.
  *Canadian Journal of Mathematics* 8:399–404, 1956.
  [doi:10.4153/CJM-1956-045-5](https://doi.org/10.4153/CJM-1956-045-5).
- <a id="din70"></a>**\[Din70\]** E. A. Dinic (Y. Dinitz). Algorithm for solution of a problem of
  maximum flow in networks with power estimation. *Soviet Mathematics Doklady* 11:1277–1280, 1970;
  Russian original *Doklady Akademii Nauk SSSR* 194(4):754–757, 1970. No DOI (verified via
  Math-Net.Ru for the original).
- <a id="cc04"></a>**\[CC04\]** E. Ciurea, L. Ciupală. Sequential and parallel algorithms for
  minimum flows. *Journal of Applied Mathematics and Computing* 15(1–2):53–75, 2004.
  [doi:10.1007/BF02935746](https://doi.org/10.1007/BF02935746).
- <a id="pic76"></a>**\[Pic76\]** J.-C. Picard. Maximal closure of a graph and applications to
  combinatorial problems. *Management Science* 22(11):1268–1272, 1976.
  [doi:10.1287/mnsc.22.11.1268](https://doi.org/10.1287/mnsc.22.11.1268).
- <a id="gk79"></a>**\[GK79\]** A. Goralčíková, V. Koubek. A reduct-and-closure algorithm for
  graphs. *MFCS 1979*, LNCS 74, pp. 301–307.
  [doi:10.1007/3-540-09526-8_27](https://doi.org/10.1007/3-540-09526-8_27).
- <a id="hmr93"></a>**\[HMR93\]** M. Habib, M. Morvan, J.-X. Rampon. On the calculation of transitive
  reduction-closure of orders. *Discrete Mathematics* 111(1–3):289–303, 1993.
  [doi:10.1016/0012-365X(93)90164-O](https://doi.org/10.1016/0012-365X(93)90164-O).
- <a id="mtkpgc19"></a>**\[MTKPGC19\]** V. Mäkinen, A. I. Tomescu, A. Kuosmanen, T. Paavilainen,
  T. Gagie, R. Chikhi. Sparse dynamic programming on DAGs with small width. *ACM Transactions on
  Algorithms* 15(2):1–21, 2019. [doi:10.1145/3301312](https://doi.org/10.1145/3301312).
- <a id="ccmrt22"></a>**\[CCMRT22\]** M. Cáceres, M. Cairo, B. Mumey, R. Rizzi, A. I. Tomescu.
  Sparsifying, shrinking and splicing for minimum path cover in parameterized linear time. *Proc.
  2022 ACM-SIAM SODA*, pp. 359–376.
  [doi:10.1137/1.9781611977073.18](https://doi.org/10.1137/1.9781611977073.18).
- <a id="ccmrt21"></a>**\[CCMRT21\]** M. Cáceres, M. Cairo, B. Mumey, R. Rizzi, A. I. Tomescu. A
  linear-time parameterized algorithm for computing the width of a DAG. *Graph-Theoretic Concepts in
  Computer Science (WG 2021)*, LNCS 12911, pp. 257–269.
  [doi:10.1007/978-3-030-86838-3_20](https://doi.org/10.1007/978-3-030-86838-3_20). Preprint
  arXiv:2007.07575.
- <a id="kp23"></a>**\[KP23\]** S. Kogan, M. Parter. Faster and unified algorithms for diameter
  reducing shortcuts and minimum chain covers. *Proc. 2023 ACM-SIAM SODA*, pp. 212–239.
  [doi:10.1137/1.9781611977554.ch9](https://doi.org/10.1137/1.9781611977554.ch9).
- <a id="bw91"></a>**\[BW91\]** G. Brightwell, P. Winkler. Counting linear extensions. *Order*
  8(3):225–242, 1991. [doi:10.1007/BF00383444](https://doi.org/10.1007/BF00383444).
- <a id="hn60"></a>**\[HN60\]** F. Harary, R. Z. Norman. Some properties of line digraphs.
  *Rendiconti del Circolo Matematico di Palermo* 9(2):161–168, 1960.
  [doi:10.1007/BF02854581](https://doi.org/10.1007/BF02854581).
- <a id="heu64"></a>**\[Heu64\]** C. Heuchenne. Sur une certaine correspondance entre graphes.
  *Bulletin de la Société Royale des Sciences de Liège* 33:743–753, 1964. No DOI. The University of
  Liège digital archive lists the article (pp. 743–753) under a later volume number; the 1964
  volume number follows the standard citation and [\[BB21\]](#bb21).
- <a id="bb21"></a>**\[BB21\]** L. W. Beineke, J. S. Bagga. *Line Graphs and Line Digraphs*.
  Developments in Mathematics 68, Springer, 2021 (chapters 10–11 on line digraphs).
  [doi:10.1007/978-3-030-81386-4](https://doi.org/10.1007/978-3-030-81386-4).
- <a id="amo93"></a>**\[AMO93\]** R. K. Ahuja, T. L. Magnanti, J. B. Orlin. *Network Flows: Theory,
  Algorithms, and Applications*. Prentice Hall, 1993. ISBN 978-0-13-617549-0 (min flow with lower
  bounds, maximum closure). No DOI.
- <a id="gt88"></a>**\[GT88\]** A. V. Goldberg, R. E. Tarjan. A new approach to the maximum-flow
  problem. *Journal of the ACM* 35(4):921–940, 1988.
  [doi:10.1145/48014.61051](https://doi.org/10.1145/48014.61051). The push–relabel method
  (`width::flow`, `--flow push-relabel`).
- <a id="hoc08"></a>**\[Hoc08\]** D. S. Hochbaum. The pseudoflow algorithm: a new algorithm for the
  maximum-flow problem. *Operations Research* 56(4):992–1009, 2008.
  [doi:10.1287/opre.1080.0524](https://doi.org/10.1287/opre.1080.0524). Deferred.

# Summary statistics

- <a id="knu98"></a>**\[Knu98\]** D. E. Knuth. *The Art of Computer Programming, Vol. 3: Sorting
  and Searching*, 2nd ed. Addison-Wesley, 1998. ISBN 0-201-89685-0. Sec. 5.3.3
  "Minimum-comparison selection": selecting the `k` largest of `n` items. `summary` keeps a
  bounded candidate buffer and compacts it by sorting, a simple variant; background only. No DOI.
  The edition, ISBN and section title are cited from the book's table of contents and were not
  re-checked against a registry in this pass.
- <a id="tmap"></a>**\[TMAP\]** The node-to-time mapping of the 2022 Payment Graph
  (per-block first node ids and block times; the half-year buckets are shipped as
  `data/chunk_boundaries.tsv`), rebuilt from the transaction list with the id rules of
  [\[PGB\]](#pgb). Node ids are chronological, and the 28 half-year chunks used to build the
  graph are contiguous id ranges (boundaries at Europe/Rome midnights); `--buckets` reads them.

# Data

- <a id="pgb"></a>**\[PGB\]** The Payment Graph builder `src/pg_el_builder.cpp` of the
  `BitcoinTemporalAnalysis` repository: one node per transaction output, numbered in chain order
  (outputs of a transaction get consecutive ids), and an arc from every resolved input of a
  transaction to each of its outputs; duplicate inputs are removed and duplicate transaction ids
  keep their first occurrence.
- <a id="bip30"></a>**\[BIP30\]** P. Wuille. BIP 30: Duplicate transactions. Bitcoin Improvement
  Proposal, 2012. <https://github.com/bitcoin/bips/blob/master/bip-0030.mediawiki>. The two
  duplicated coinbase transactions (heights 91,842 and 91,880) are the only duplicate ids.
- <a id="cve-2018-17144"></a>**\[CVE-2018-17144\]** Bitcoin Core. CVE-2018-17144 full disclosure
  (duplicate inputs in one transaction), 2018.
  <https://bitcoincore.org/en/2018/09/20/notice/>; NVD entry
  <https://nvd.nist.gov/vuln/detail/CVE-2018-17144>. Duplicate inputs are invalid, so a node has at
  most one spender.
