
#include "dataTypes.h"
#include <iostream>
#include <math.h>

using namespace PGMplusplus;


int main (int argc, char* argv[]) {



    CGraph myGraph;

    // Desired result
    // tests.Y1 = [ 5 1 2 3 3 7 2 2 6 1 4 ];

    // General configuration

    // Feature multipliers

    Eigen::VectorXd nodeFeatMultiplierFactor(5);
    nodeFeatMultiplierFactor << 50, 90, 60, 50, 50;

    Eigen::VectorXd edgeFeatMultiplierFactor(4);
    edgeFeatMultiplierFactor << 1, 1, 50, 1;

    // Edge features to use

    Eigen::VectorXi edgeFeaturesToUse(6);
    edgeFeaturesToUse << 1, 1, 0, 0, 1, 1;


    /*--------------------------------------------------------------------------
     *
     *                                  NODES
     *
     *-------------------------------------------------------------------------*/

    Eigen::Matrix<double,11,5> node_features;


    // Features are: orientation centroid_z area elongation bias

    node_features << 1, 0.725256, 0.347833, 2.32114, 1,
            0, 0, 1.46145, 2.86637, 1,
            1, 0.513579, 0.398025, 1.85308, 1,
            0, 0.896507, 0.969337, 1.7035, 1,
            0, 0.748331, 0.585186, 1.47532, 1,
            1, 0.963636, 0.125714, 1, 1,
            1, 0.93444, 0.212592, 1.52702, 1,
            1, 1.05757, 0.236803, 3.22358, 1,
            0, 0.398408, 0.131399, 1, 1,
            0, 0.0300602, 0.129521, 2.03183, 1,
            1, 0.469344, 0.505431, 1.45506, 1;

    size_t N_nodes = 11;
    size_t N_features = 4;

    for ( size_t i = 0; i < N_nodes; i++ )
    {
        Eigen::VectorXd features = node_features.row(i);

        //std::cout << "Features before: " << features << std::endl;

        Eigen::VectorXd scaledFeatures =
                features = features.cwiseProduct( nodeFeatMultiplierFactor );

        //std::cout << "Features after: " << scaledFeatures << std::endl;

        CNode node( 0, // type
                    scaledFeatures // node features
                    //features
                    );

        myGraph.addNode(node);

        //std::cout << node << std::endl;
    }


    /*--------------------------------------------------------------------------
     *
     *                                  EDGES
     *
     *-------------------------------------------------------------------------*/

    Eigen::Matrix<float,28,2> edges;
    edges << 1, 2,
    1, 3,
    1, 4,
    1, 5,
    1, 6,
    1, 9,
    1, 10,
    2, 3,
    2, 9,
    2, 10,
    4, 5,
    4, 6,
    4, 7,
    4, 8,
    4, 9,
    4, 11,
    5, 6,
    5, 7,
    5, 8,
    5, 9,
    5, 10,
    5, 11,
    6, 7,
    7, 8,
    7, 9,
    7, 11,
    8, 11,
    9, 10;

    Eigen::Matrix<int,28,1> pre_computed_edge_features;

    pre_computed_edge_features << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    size_t N_edgeFeatures = edgeFeaturesToUse.sum();

    // Create edges

    size_t N_edges = 28;

    for ( size_t i = 0; i < N_edges; i++ )
    {
        Eigen::Vector2f nodes = edges.row(i);
        Eigen::Vector2f ones(1,1);
        nodes = nodes - ones;

        CNode n1 = myGraph.getNode(nodes(0));
        CNode n2 = myGraph.getNode(nodes(1));

        CEdge edge(n1,n2,0);

        Eigen::Matrix<double,Eigen::Dynamic,1> edgeFeatures;
        edgeFeatures.resize(N_edgeFeatures);

        Eigen::VectorXd feat1 = n1.getFeatures();
        Eigen::VectorXd feat2 = n2.getFeatures();

        // Computation of the edge features;

        size_t feature = 0;


        // Perpendicularity
        if ( edgeFeaturesToUse(0) )
        {
            edgeFeatures(feature) = std::abs(feat1(0) - feat2(0));
            feature++;
        }

        // Height distance between centers
        if ( edgeFeaturesToUse(1) )
        {
            edgeFeatures(feature) = std::abs((float)feat1(1) - (float)feat2(1));
            feature++;
        }

        // Ratio between areas
        if ( edgeFeaturesToUse(2) )
        {
            float a1 = feat1(2);
            float a2 = feat2(2);

            if ( a1 < a2 )
            {
                float aux = a2;
                a2 = a1;
                a1 = aux;
            }

            edgeFeatures(feature) = a1 / a2;
            feature++;
        }

        //  Difference between elongations
        if  ( edgeFeaturesToUse(3) )
        {
            edgeFeatures(feature) = std::abs(feat1(3) - feat2(3));
            feature++;
        }

        // Use the isOn semantic relation?

        if ( edgeFeaturesToUse(4) )
        {
            edgeFeatures(feature) = pre_computed_edge_features(i);
            feature++;
        }

        // Use bias feature in edges?
        if ( edgeFeaturesToUse(5) )
        {
            edgeFeatures(feature) = 1;
            feature++;
        }


        Eigen::VectorXd scaledFeatures =
                 edgeFeatures.cwiseProduct( edgeFeatMultiplierFactor );

        edge.setFeatures( scaledFeatures );

        std::cout << "----" << std::endl << scaledFeatures << std::endl;

        myGraph.addEdge( edge );
    }


    /*--------------------------------------------------------------------------
     *
     *                                  WEIGHTS
     *
     *-------------------------------------------------------------------------*/

    Eigen::Matrix<double,372,1> w;

//    size_t N = 372;
//    w.resize(N,1);

    w << 0.0147619427138712,
        0.0173616109979809,
       -0.0247242551197866,
        0.0134129547356221,
        0.0112902099822376,
        -0.0337066365384854,
        0.00776209990358690,
        -0.00336497714745827,
        0.00358188214638315,
        0.00134973197353300,
        0.0239678188818275,
        -0.00216849710156958,
        -0.0564461634933869,
        -0.00157231575892016,
        0.0225830248603814,
        -0.0296688084468314,
        0.00243456460313082,
        -0.00521049039815414,
        0.0303925293339997,
        -0.00743192546212200,
        0.000225488284480376,
        0.0153939701987018,
        0.0332643467413878,
        -0.00396422046266730,
        0.0325748911708760,
        0.0371835203396646,
        0.0292332708559039,
        -0.0247914096738959,
        -0.0372935583001342,
        -0.0275863653517021,
        0.00794151489984904,
        0.000259236609674822,
        -0.0410399735737298,
        -0.00296060268378853,
        0.0276545474611011,
        -0.00117507175381891,
        0.0197029782990930,
        0.0196557622081615,
        -0.000466289201071175,
        0.00565705283769277,
        0.00132121299902517,
        -0.00403657556634580,
        -0.0402280971479811,
        -0.00754122003102943,
        0.0107225503132089,
        0.0162776317227574,
        -0.0161301087715455,
        -0.00493489766196552,
        0.0138757813565257,
        -0.0347520164004827,
        -0.00177594861571483,
        0.0347179113418672,
        0.0347150941914294,
        0.0707669390303767,
        0.0316004511388923,
        -0.0131070454657615,
        -0.00679304787658545,
        -0.0435442631432814,
        -0.0547614336571494,
        -0.0309424219001161,
        -0.00552691426176521,
        0.0111225838308921,
        -0.00751886750437491,
        0.00958585551526243,
        0.00812629465170327,
        -0.0181425423554257,
        -0.00592491794043658,
        -0.000915122562072345,
        -0.00742903405474238,
        -0.00773810364068306,
        -0.00104061440307466,
        -0.000486224074897712,
        -0.0178856841350399,
        0.0107693140186158,
        -0.0157662867951366,
        -0.0158576612666365,
        0.00123379857458252,
        -0.0157005569996267,
        0.000642848836062395,
        -0.00428148962076404,
        -0.00472480186834645,
        0.0127829298083490,
        -0.000415527254561359,
        -0.0120325714904446,
        0.0200685202320895,
        0.0243352534025739,
        -0.0242383173551293,
        0.00213076088816421,
        -0.00216478844512440,
        0.00905099816056925,
        0.00601926751282504,
        0.00168184779271244,
        -0.00139621150728500,
        -0.00206650855598336,
        -0.00891587822321930,
        0.0229844512940923,
        -0.00735663220074516,
        0.00318341287264961,
        -0.00166071624015877,
        -0.00178241909067436,
        0.00817229821355887,
        -0.000112932448406294,
        -0.00252668324495422,
        0.0268933840910119,
        -0.00501358343591918,
        -0.00138714977303632,
        -0.00209666202790933,
        -0.00276973380747935,
        0.00354891915877519,
        -0.000359651929107668,
        -0.00799066815647181,
        0.00205933584930208,
        -0.00141576566232497,
        0.0118706716517969,
        0.00397719441388296,
        0.00149892357805191,
        -0.000770744757244464,
        -0.00188774694077132,
        0.00364846935231804,
        -0.00212246801250505,
        -0.00170055870393622,
        0.00879273663255836,
        -0.000185562459664041,
        -0.00000244716207280937,
        0.00154539992846604,
        -0.000481917556687306,
        0.00330382487913055,
        -0.00000111790187099125,
        -0.000128055628305863,
        -0.000363128626056712,
        0.00261756056235542,
        -0.0000279389301935642,
        -0.000269275865253509,
        0.00186884404154169,
        -0.0000292014684960087,
        -0.000878405454471036,
        -0.0000259079444072247,
        0,
        -0.00844118538953171,
        0.0175504920956733,
        -0.0179832809673411,
        0.000511423721150691,
        0.0170054061594603,
        0.0267704710132272,
        -0.00661151182572854,
        -0.00182165636943345,
        -0.00931454873056065,
        -0.0120508309507082,
        -0.00131994242070125,
        -0.000891946927291174,
        0.000882261124317558,
        0.0143287906250534,
        0.0149068985881391,
        0.00800480706966862,
        0.00946825836847495,
        0.00912524043978500,
        0.00177951742366339,
        0.0114647336451091,
        0.00736383961733324,
        0.0118027374969430,
        -0.000898539341389845,
        -0.0160814756529628,
        0.00127552260191487,
        -0.0484786809583628,
        0.0162478801357332,
        0.000304239398073502,
        0.00321644367302266,
        -0.0132518487541618,
        0.00487336530877702,
        0.00706479689951252,
        -0.000759288120811617,
        -0.0105530181964375,
        0.00782137382343906,
        -0.0318206325749236,
        0.0150774631162713,
        0.00358711317316208,
        0.00786282284071955,
        0.0117519559074773,
        0.0124547439354561,
        -0.000223664280820788,
        -0.0434309598333362,
        -0.00823741222426154,
        -0.00495339361226253,
        -0.00177530554951495,
        -0.00463912910298116,
        -0.000373234276530487,
        0.000539510081533958,
        -0.000475284774031804,
        -0.0117659706547845,
        0.00786430645920024,
        -0.000863645337734455,
        0.00382068747921586,
        0.00946672477640742,
        0.00596655088478059,
        -0.000494190174040713,
        -0.0221213281253105,
        0.00764511221428522,
        0.00257342843661586,
        -0.00612430542444291,
        0.00158584466730420,
        -0.000510523483422908,
        -0.00000380430821997208,
        0.00365838048326983,
        -0.00113132133383439,
        0.00243634096713345,
        -0.00000190807272658544,
        -0.00361669232903752,
        -0.000645423249849568,
        0.00164939528159750,
        -0.0000798221344639696,
        -0.00296694676035164,
        0.00651278478006863,
        -0.0000916152186489369,
        -0.00137347166056982,
        -0.0000439256114468751,
        0,
        0,
        0,
        -0.000411560432724252,
        0,
        0,
        0,
        -0.00299639430272139,
        0,
        -0.000188821668548181,
        -0.000304541661333010,
        0,
        0,
        0,
        -0.0284634024133441,
        0,
        0,
        0,
        -0.000977422389295253,
        0,
        -0.0000600932784611612,
        -0.000174449676034357,
        0,
        0,
        -0.00554083392303780,
        -0.0152001677179349,
        -0.0325928024997786,
        -0.00421516317804539,
        0.0474139581674230,
        -0.000898085574694195,
        0.0390984124074465,
        0.0333082244301199,
        -0.0168931808307743,
        -0.000564335339849830,
        0,
        0,
        0,
        -0.0000424823158160715,
        0,
        -0.00000523415269665126,
        -0.00000779681052657593,
        0,
        0,
        0,
        0,
        -0.0000953124475836707,
        0,
        -0.0000110212436215370,
        -0.0000157870955749595,
        0,
        0,
        0,
        -0.00506733646218016,
        0,
        -0.000788689728812163,
        -0.000794139734993718,
        0,
        0,
        -0.000305385342800444,
        -0.0000881666171999063,
        -0.000101418803116067,
        -0.000422311077040574,
        -0.00202530816238699,
        -0.0000319865353570155,
        0,
        -0.00000598086527625661,
        -0.00000882700868459060,
        0,
        0,
        -0.00000711991746147508,
        -0.0000399082457754043,
        -0.000130587280269784,
        -0.00000249111621302259,
        -0.00000537847078063945,
        -0.000283882653560529,
        -0.00000438179365865572,
        0,
        0,
        0,
        0.000679539326912305,
        -0.00105733698196059,
        -0.00967553181884249,
        0.000464054448226658,
        -0.00202752906805321,
        0.00767568188155505,
        -0.00322808409805447,
        -0.000559169976782304,
        -0.00284224089362016,
        -0.00286022312183879,
        -0.00222312195169412,
        -0.000255929599612073,
        -0.0268293705880230,
        -0.00128620164849102,
        0.0145875867725839,
        0.0289106840717731,
        -0.00791242381322802,
        0.00235328604354677,
        -0.000262137041234777,
        0.00422688695963269,
        0.0104294398348749,
        0.000685814949008684,
        -0.000458767858087036,
        -0.00809871415123701,
        0.00349954029772845,
        0.00534685846184118,
        0.0168018054026761,
        -0.00172831673149324,
        -0.000135636953172746,
        0.00176342217393236,
        0.000704195811712654,
        -0.00351593999466683,
        -0.000554926882975266,
        -0.00179278223360503,
        0.0223676988904071,
        0.00631509320754007,
        0.00365680279462862,
        0.000944987178642110,
        0.00367222386908539,
        0.00104603605578649,
        0.00160388184658270,
        -0.000126746700226290,
        -0.0442091331050918,
        0.00542067524612832,
        -0.00479572663622171,
        -0.000825900904712862,
        0.00516471755316996,
        -0.000800148478788366,
        -0.00567463308841988,
        -0.000314379330259199,
        -0.00673127850427281,
        -0.00318136146922850,
        -0.000509547176714300,
        0.00275553834331088,
        -0.0000939651962585357,
        -0.00363530290477947,
        -0.000269665297721591,
        -0.0491006427620047,
        0.000928535667236849,
        0.0282336910414986,
        0.0276330645127431,
        -0.00146081151266117,
        -0.000253933294572408,
        -0.000000809285923883426,
        0.000538179127391408,
        -0.000265155638092349,
        0.000624464268995357,
        -0.000000381923631882622,
        -0.00608920622227450,
        0.00323266360410324,
        0.00136696632118849,
        -0.0000535529461449478,
        -0.00725942302115148,
        -0.000155051040317261,
        -0.0000572153196319327,
        -0.000454236198193242,
        -0.0000114226004761813,
        0;


    Eigen::Matrix<int,5,12> nodeMap;

    nodeMap << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
            49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60;

    Eigen::MatrixXd nodeWeights;

    size_t N_rows = nodeMap.cols();
    size_t N_cols = nodeMap.rows();

    nodeWeights.resize(N_rows,N_cols);

    for ( size_t i = 0; i < N_rows; i++ )
        for ( size_t j = 0; j < N_cols; j++ )
        {
            nodeWeights(i,j) = w(nodeMap(j,i)-1);
        }

    myGraph.addNodeWeights( nodeWeights );

    //std::cout << nodeWeights << std::endl;

//    std::vector<Eigen::Array<Eigen::MatrixXd,Eigen::Dynamic,Eigen::Dynamic> > m_edgeWeights;



    size_t f = 61;
    size_t first_f = 61;

    size_t N_classes = nodeWeights.rows();

    std::vector<Eigen::MatrixXd> edgeWeights;

    edgeWeights.resize(N_edgeFeatures);



    for (size_t i = 0; i < N_edgeFeatures; i++ )
        edgeWeights[i].resize(N_classes,N_classes);

    for ( size_t edge_feat = 0; edge_feat < N_edgeFeatures; edge_feat++ )
    {

        for ( size_t c1 = 0; c1 < N_classes; c1++ ) // rows
        {
            f++;

            for ( size_t c2 = c1+1; c2 < N_classes; c2++ )  // cols
            {
                     edgeWeights[edge_feat](c1,c2) = f;
                     edgeWeights[edge_feat](c2,c1) = f;
                     f++;
                     //std::cout << "f: " << f << std::endl;
            }
        }

            size_t previousW = first_f;

            edgeWeights[edge_feat](0,0) = first_f;

            //  edgeMap(1,1,:,edgeFeat) = firstW;

            for ( size_t c3 = 1; c3 < N_classes; c3++ )
            {
                f = previousW + (N_classes - c3) + 1;

                //std::cout << "F: " << f << std::endl;
                edgeWeights[edge_feat](c3,c3) = f;
                previousW = f;
            }

            first_f = edgeWeights[edge_feat](N_classes-1,N_classes-1)+1;
            f++;

    }

    // Replace indexes by their values in w

    for ( size_t edge_feat = 0; edge_feat < N_edgeFeatures; edge_feat++ )
        for ( size_t c1 = 0; c1 < N_classes; c1++ ) // rows
            for ( size_t c2 = 0; c2 < N_classes; c2++ )  // cols
            {
                edgeWeights[edge_feat](c1,c2) = w(edgeWeights[edge_feat](c1,c2)-1);
            }

    //std::cout << "Edge weights 0:" << std::endl << edgeWeights[0] << std::endl;
    //std::cout << "Edge weights 1:" << std::endl << edgeWeights[1] << std::endl;
    //std::cout << "Edge weights 2:" << std::endl << edgeWeights[2] << std::endl;
    //std::cout << "Edge weights 1:" << std::endl << edgeWeights[3] << std::endl;

    myGraph.addEdgeWeights( edgeWeights );

    myGraph.computePotentials();
    //std::cout << myGraph << std::endl;

    std::vector<size_t> results;

    myGraph.decodeICM(results);

    std::cout << "RESULTS ICM! " << std::endl;

    for ( size_t i = 0; i < results.size(); i++ )
    {
        std::cout << results[i] << std::endl;
    }


    myGraph.decodeGreedy(results);

    std::cout << "RESULTS Greedy! " << std::endl;

    for ( size_t i = 0; i < results.size(); i++ )
    {
        std::cout << results[i] << std::endl;
    }


    return 1;
}
