// #include "GeneratePartition.h"

// #include "Error.h"

// namespace nacho {

// using llir::lExpr;
// using llir::lStmt;

// namespace {

// // Base case.
// llir::Function generate_binary_1d_partition(const std::string &a,
//                                             const std::string &b) {
//     llir::Function func;
//     static const llir::lType index_t = llir::Generic_t::make("index_t");
//     func.generics = {"index_t"};
//     // TODO: set device attribute.
//     func.ret_type = llir::Tuple_t::make({index_t, index_t});
//     func.name = "partition2D";
//     static const llir::lType ptr_t = llir::Ptr_t::make(index_t);
//     // Set arguments
//     for (const auto &e : {a, b}) {
//         func.args.emplace_back(llir::Function::Argument{
//             .mutating = false, .type = ptr_t, .name = e + "_crd"});
//         func.args.emplace_back(llir::Function::Argument{
//             .mutating = false, .type = index_t, .name = e + "_nnz"});
//     }
//     // Partition value
//     func.args.emplace_back(llir::Function::Argument{
//         .mutating = false, .type = index_t, .name = "count"});

//     // The body is exactly what we would hand-write, the interesting codegen is
//     // in the non base case below

//     std::vector<lStmt> stmts;

//     stmts.emplace_back(
//         // TODO: make this not start at 0!
//         // index_t a_low = 0;
//         llir::Declare::make(index_t, a + "_low",
//                             llir::lConst::make((int64_t)0)));

//     lExpr a_nnz = llir::lVar::make(index_t, a + "_nnz");
//     lExpr b_nnz = llir::lVar::make(index_t, b + "_nnz");
//     lExpr count = llir::lVar::make(index_t, "count");
//     stmts.emplace_back(
//         // TODO: make this start at min(count, a_nnz)
//         // index_t a_high = count;
//         llir::Declare::make(index_t, a + "_high", count));

//     std::vector<lStmt> stmts_inside_while;

//     lExpr a_low = llir::lVar::make(index_t, a + "_low");
//     lExpr a_high = llir::lVar::make(index_t, a + "_high");
//     stmts_inside_while.emplace_back(
//         // index_t a_mid = a_low + (a_high - a_low) / 2;
//         llir::Declare::make(index_t, a + "_mid",
//                             a_low + (a_high - a_low) / llir::lConst::make(2)));

//     lExpr a_mid = llir::lVar::make(index_t, a + "_mid");
//     stmts_inside_while.emplace_back(
//         // index_t a_mid = a_low + (a_high - a_low) / 2;
//         llir::Declare::make(index_t, b + "_choice", count - a_mid));
//     lExpr b_choice = llir::lVar::make(index_t, b + "_choice");

//     // Useful repeated constants
//     static const lExpr _0 = llir::lConst::make(0);
//     static const lExpr _1 = llir::lConst::make(1);

//     lExpr A = llir::lVar::make(ptr_t, a);
//     lExpr B = llir::lVar::make(ptr_t, b);

//     // (l_b > 0 && l_a < nA && B[l_b - 1] > A[l_a])
//     lExpr am_too_small =
//         (b_choice > _0 && a_mid < a_nnz && B[b_choice - _1] > A[a_mid]);
//     // (l_a > 0 && l_b < nB && A[l_a - 1] > B[l_b])
//     lExpr am_too_big =
//         (a_mid > _0 && b_choice < b_nnz && A[a_mid - _1] > B[b_choice]);

//     lStmt check = llir::IfElse::make(
//         am_too_small,
//         // low = l_a + 1;
//         llir::Store::make(a + "_low", a_mid + _1),
//         llir::IfElse::make(am_too_big,
//                            // high = l_a;
//                            llir::Store::make(a + "_high", a_mid),
//                            llir::Return::make(llir::lBuild::make(
//                                func.ret_type, {a_mid, b_choice}))));

//     stmts_inside_while.emplace_back(std::move(check));

//     lStmt while_loop = llir::While::make(
//         a_low < a_high, llir::Sequence::make(std::move(stmts_inside_while)));

//     stmts.emplace_back(std::move(while_loop));

//     // Add return at the end.
//     // index_t l_a = low;
//     // index_t l_b = count - l_a;
//     // return std::make_tuple(l_a, l_b);
//     stmts.emplace_back(llir::Return::make(
//         llir::lBuild::make(func.ret_type, {a_low, count - a_low})));
//     func.body = llir::Sequence::make(std::move(stmts));

//     return func;
// }

// } // namespace

// std::vector<llir::Function>
// generate_nary_1d_partition(const std::vector<std::string> &names) {
//     internal_assert(names.size() >= 2)
//         << "Cannot generate partition function for: " << names.size()
//         << " values (requires >= 2)";
    
//     if (names.size() == 2) {
//         // Base case
//         return {generate_binary_1d_partition(names[0], names[1])};
//     } else {
//         internal_error << "TODO: handle non-base case for partition generation.";
//     }
// }

// } // namespace nacho
